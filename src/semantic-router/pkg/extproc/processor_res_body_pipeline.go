package extproc

import (
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (r *OpenAIRouter) handleNonStreamingResponseBody(
	responseBody []byte,
	ctx *RequestContext,
	completionLatency time.Duration,
	initialBodyTransformed bool,
) *ext_proc.ProcessingResponse {
	usage := parseResponseUsage(responseBody, ctx.RequestModel)
	r.reportNonStreamingUsage(ctx, completionLatency, usage)
	r.calibrateTokenEstimator(ctx, usage.promptTokens)
	r.updateResponseCache(ctx, responseBody)

	finalBody, clientTransformed := r.translateResponseBodyForClient(ctx, responseBody)
	bodyMutation, headerMutation := buildInitialResponseMutations(
		finalBody,
		initialBodyTransformed || clientTransformed,
	)

	if jailbreakResponse := r.performResponseJailbreakDetection(ctx, responseBody); jailbreakResponse != nil {
		return jailbreakResponse
	}
	if hallucinationResponse := r.performHallucinationDetection(ctx, responseBody); hallucinationResponse != nil {
		return hallucinationResponse
	}

	r.scheduleResponseMemoryStore(ctx, responseBody)
	r.markUnverifiedFactualResponse(ctx)

	response := r.applyResponseWarnings(ctx, responseBody, bodyMutation, headerMutation)
	r.applyResponseCost(response, ctx.ResponseCost, finalBody)
	r.updateRouterReplayHallucinationStatus(ctx)
	r.attachRouterReplayResponse(ctx, finalBody, true)
	return response
}

// translateResponseBodyForClient dispatches body rewriting based on the
// inbound client protocol. Returns the rewritten body and a flag
// indicating whether a rewrite happened, so callers know whether to
// emit a body mutation downstream.
//
// Branch order matters: ClientProtocol "anthropic" and Response API are
// mutually exclusive (different inbound paths — /v1/messages vs
// /v1/responses), but pinning the Anthropic branch first keeps the
// dispatcher reading top-down by likelihood.
func (r *OpenAIRouter) translateResponseBodyForClient(
	ctx *RequestContext,
	responseBody []byte,
) ([]byte, bool) {
	// Defensive: a nil ctx cannot match either dispatcher branch and the
	// downstream isResponseAPIRequest dereference would panic. Treat as
	// plain OpenAI pass-through.
	if ctx == nil {
		return responseBody, false
	}

	if ctx.ClientProtocol == config.ClientProtocolAnthropic {
		translated, err := anthropic.EmitAnthropicResponse(responseBody, ctx.IRExtensions, ctx.RequestModel)
		if err != nil {
			logging.Errorf("Anthropic outbound emit failed: %v", err)
			return anthropic.EmitAnthropicError("api_error", err.Error()), true
		}
		return translated, true
	}

	if !isResponseAPIRequest(ctx) || r.ResponseAPIFilter == nil {
		return responseBody, false
	}

	translatedBody, err := r.ResponseAPIFilter.TranslateResponse(
		ctx.TraceContext,
		ctx.ResponseAPICtx,
		responseBody,
	)
	if err != nil {
		logging.Errorf("Response API translation error: %v", err)
		return responseBody, false
	}

	logging.Infof("Response API: Translated response to Response API format")
	return translatedBody, true
}

func buildInitialResponseMutations(
	finalBody []byte,
	bodyWasTransformed bool,
) (*ext_proc.BodyMutation, *ext_proc.HeaderMutation) {
	if !bodyWasTransformed {
		return nil, nil
	}

	return &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_Body{
				Body: finalBody,
			},
		}, &ext_proc.HeaderMutation{
			RemoveHeaders: []string{"content-length"},
		}
}

func (r *OpenAIRouter) markUnverifiedFactualResponse(ctx *RequestContext) {
	if ctx.VSRSelectedDecision == nil {
		return
	}

	hallucinationConfig := ctx.VSRSelectedDecision.GetHallucinationConfig()
	if hallucinationConfig != nil && hallucinationConfig.Enabled {
		r.checkUnverifiedFactualResponse(ctx)
	}
}

// applyResponseWarnings consolidates the response-quality warnings into the
// single x-vsr-response-warnings header (#2204). Each warning either contributes
// its code to that header (the "header"/default action), rewrites the body with
// an inline warning (the "body" action), or is suppressed ("none"). Codes are
// collected in a fixed order so the header value is deterministic. The detail
// behind each warning stays in the replay record.
func (r *OpenAIRouter) applyResponseWarnings(
	ctx *RequestContext,
	responseBody []byte,
	bodyMutation *ext_proc.BodyMutation,
	headerMutation *ext_proc.HeaderMutation,
) *ext_proc.ProcessingResponse {
	response := buildResponseBodyContinueResponse(bodyMutation, headerMutation)
	modifiedBody := responseBody
	var codes []string

	if ctx.HallucinationDetected {
		var code string
		modifiedBody, code = r.applyHallucinationWarning(ctx, modifiedBody)
		codes = appendNonEmpty(codes, code)
	}
	if ctx.UnverifiedFactualResponse {
		var code string
		modifiedBody, code = r.applyUnverifiedFactualWarning(ctx, modifiedBody)
		codes = appendNonEmpty(codes, code)
	}
	// Jailbreak never rewrites the body, so its position relative to the
	// body-prepending warnings above is immaterial; it is appended last only to
	// fix the code order in the header value.
	codes = appendNonEmpty(codes, r.responseJailbreakWarningCode(ctx))

	if len(codes) > 0 {
		setResponseWarningsHeader(response, codes)
	}
	if string(modifiedBody) != string(responseBody) {
		setResponseBodyMutation(response, modifiedBody)
	}
	return response
}

func appendNonEmpty(codes []string, code string) []string {
	if code == "" {
		return codes
	}
	return append(codes, code)
}

// setResponseWarningsHeader writes the consolidated x-vsr-response-warnings header
// (comma-separated codes) onto the response, merging with any existing mutation.
func setResponseWarningsHeader(response *ext_proc.ProcessingResponse, codes []string) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.Response == nil {
		bodyResponse.ResponseBody.Response = &ext_proc.CommonResponse{}
	}
	opt := &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      headers.VSRResponseWarnings,
			RawValue: []byte(strings.Join(codes, ",")),
		},
	}
	if hm := bodyResponse.ResponseBody.Response.HeaderMutation; hm != nil {
		hm.SetHeaders = append(hm.SetHeaders, opt)
		return
	}
	bodyResponse.ResponseBody.Response.HeaderMutation = &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{opt},
	}
}

func setResponseBodyMutation(response *ext_proc.ProcessingResponse, body []byte) {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	bodyResponse.ResponseBody.Response.BodyMutation = &ext_proc.BodyMutation{
		Mutation: &ext_proc.BodyMutation_Body{
			Body: body,
		},
	}
}

// applyResponseCost returns the consolidated request cost to the client on the
// non-streaming path: the x-vsr-response-cost* headers plus a cost block inside
// the body usage object. It runs last so it injects into whatever body the
// warning/translation steps settled on, falling back to fallbackBody when no
// mutation was set (plain upstream pass-through). No-op when cost is nil.
func (r *OpenAIRouter) applyResponseCost(
	response *ext_proc.ProcessingResponse,
	cost *responseCost,
	fallbackBody []byte,
) {
	if cost == nil {
		return
	}
	setResponseCostHeaders(response, cost)

	body, ok := currentResponseBody(response)
	if !ok {
		body = fallbackBody
	}
	injected, changed := injectCostIntoUsage(body, cost)
	if !changed {
		return
	}
	setResponseBodyMutation(response, injected)
	ensureContentLengthRemoved(response)
}

// setResponseCostHeaders writes the three x-vsr-response-cost* headers onto the
// response, merging with any existing header mutation (warnings, content-length).
func setResponseCostHeaders(response *ext_proc.ProcessingResponse, cost *responseCost) {
	appendResponseHeaders(response, []*core.HeaderValueOption{
		{Header: &core.HeaderValue{Key: headers.VSRResponseCost, RawValue: []byte(formatCost(cost.Total))}},
		{Header: &core.HeaderValue{Key: headers.VSRResponseCostCurrency, RawValue: []byte(cost.Currency)}},
		{Header: &core.HeaderValue{Key: headers.VSRResponseCostBreakdown, RawValue: []byte(cost.breakdownHeaderValue())}},
	})
}

// appendResponseHeaders adds set-header options to the response's header mutation,
// creating the mutation when absent.
func appendResponseHeaders(response *ext_proc.ProcessingResponse, opts []*core.HeaderValueOption) {
	common := commonResponseFor(response)
	if common == nil {
		return
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	common.HeaderMutation.SetHeaders = append(common.HeaderMutation.SetHeaders, opts...)
}

// ensureContentLengthRemoved guarantees the upstream content-length is dropped so
// Envoy recomputes it for the mutated body; it is idempotent.
func ensureContentLengthRemoved(response *ext_proc.ProcessingResponse) {
	common := commonResponseFor(response)
	if common == nil {
		return
	}
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	for _, h := range common.HeaderMutation.RemoveHeaders {
		if h == "content-length" {
			return
		}
	}
	common.HeaderMutation.RemoveHeaders = append(common.HeaderMutation.RemoveHeaders, "content-length")
}

// currentResponseBody returns the body already staged in the response's body
// mutation, if any.
func currentResponseBody(response *ext_proc.ProcessingResponse) ([]byte, bool) {
	common := commonResponseFor(response)
	if common == nil || common.BodyMutation == nil {
		return nil, false
	}
	if b, ok := common.BodyMutation.Mutation.(*ext_proc.BodyMutation_Body); ok {
		return b.Body, true
	}
	return nil, false
}

// commonResponseFor extracts the CommonResponse from a response-body processing
// response, returning nil for any other shape.
func commonResponseFor(response *ext_proc.ProcessingResponse) *ext_proc.CommonResponse {
	bodyResponse, ok := response.Response.(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return nil
	}
	return bodyResponse.ResponseBody.Response
}

func isResponseAPIRequest(ctx *RequestContext) bool {
	return ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.IsResponseAPIRequest
}
