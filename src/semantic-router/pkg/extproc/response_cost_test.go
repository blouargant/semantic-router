/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func costTestRouter() *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"premium":  {Pricing: config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 30, Currency: "USD"}},
				"balanced": {Pricing: config.ModelPricing{PromptPer1M: 1, CompletionPer1M: 3, Currency: "USD"}},
				"free":     {}, // no pricing configured
			},
		},
	}}
}

func TestBuildResponseCost_PerModelAndTotal(t *testing.T) {
	r := costTestRouter()
	cost := r.buildResponseCost([]costModelLeg{
		{model: "premium", usage: responseUsageMetrics{promptTokens: 1000, completionTokens: 1000}},
		{model: "balanced", usage: responseUsageMetrics{promptTokens: 1000, completionTokens: 1000}},
	})
	if cost == nil {
		t.Fatal("expected non-nil cost report")
	}
	// premium: (1000*10 + 1000*30)/1e6 = 0.04; balanced: (1000*1 + 1000*3)/1e6 = 0.004
	if cost.Total != 0.044 {
		t.Errorf("Total = %v, want 0.044", cost.Total)
	}
	if cost.Currency != "USD" {
		t.Errorf("Currency = %q, want USD", cost.Currency)
	}
	if len(cost.PerModel) != 2 || cost.PerModel[0].Model != "premium" || cost.PerModel[1].Model != "balanced" {
		t.Fatalf("PerModel = %+v, want premium then balanced", cost.PerModel)
	}
	if cost.PerModel[0].Cost != 0.04 || cost.PerModel[1].Cost != 0.004 {
		t.Errorf("per-model costs = %v/%v, want 0.04/0.004", cost.PerModel[0].Cost, cost.PerModel[1].Cost)
	}
}

func TestBuildResponseCost_UnpricedYieldsNil(t *testing.T) {
	r := costTestRouter()
	if cost := r.buildResponseCost([]costModelLeg{{model: "free", usage: responseUsageMetrics{promptTokens: 100, completionTokens: 100}}}); cost != nil {
		t.Errorf("expected nil for unpriced model, got %+v", cost)
	}
	if cost := r.buildResponseCost([]costModelLeg{{model: "unknown", usage: responseUsageMetrics{promptTokens: 100}}}); cost != nil {
		t.Errorf("expected nil for unknown model, got %+v", cost)
	}
}

func TestBuildResponseCost_SkipsUnpricedLegsButKeepsPriced(t *testing.T) {
	r := costTestRouter()
	cost := r.buildResponseCost([]costModelLeg{
		{model: "free", usage: responseUsageMetrics{promptTokens: 1000, completionTokens: 1000}},
		{model: "balanced", usage: responseUsageMetrics{promptTokens: 1000, completionTokens: 1000}},
	})
	if cost == nil || len(cost.PerModel) != 1 || cost.PerModel[0].Model != "balanced" {
		t.Fatalf("expected only the priced balanced leg, got %+v", cost)
	}
}

func TestInjectCostIntoUsage(t *testing.T) {
	cost := &responseCost{Total: 0.044, Currency: "USD", PerModel: []modelCostEntry{{Model: "premium", Cost: 0.04}}}

	t.Run("adds cost block to usage object", func(t *testing.T) {
		body := []byte(`{"object":"chat.completion","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
		out, ok := injectCostIntoUsage(body, cost)
		if !ok {
			t.Fatal("expected injection to succeed")
		}
		var parsed struct {
			Usage map[string]interface{} `json:"usage"`
		}
		if err := json.Unmarshal(out, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if parsed.Usage["cost"].(float64) != 0.044 {
			t.Errorf("usage.cost = %v, want 0.044", parsed.Usage["cost"])
		}
		if parsed.Usage["currency"].(string) != "USD" {
			t.Errorf("usage.currency = %v, want USD", parsed.Usage["currency"])
		}
		if parsed.Usage["cost_per_model"] == nil {
			t.Error("usage.cost_per_model missing")
		}
		// Original token counts survive.
		if parsed.Usage["prompt_tokens"].(float64) != 10 {
			t.Errorf("prompt_tokens overwritten: %v", parsed.Usage["prompt_tokens"])
		}
	})

	t.Run("no usage object leaves body unchanged", func(t *testing.T) {
		body := []byte(`{"object":"chat.completion","choices":[]}`)
		out, ok := injectCostIntoUsage(body, cost)
		if ok || string(out) != string(body) {
			t.Errorf("expected no-op for body without usage, got ok=%v body=%s", ok, out)
		}
	})

	t.Run("non-json leaves body unchanged", func(t *testing.T) {
		body := []byte(`not json`)
		out, ok := injectCostIntoUsage(body, cost)
		if ok || string(out) != string(body) {
			t.Errorf("expected no-op for non-json, got ok=%v", ok)
		}
	})

	t.Run("nil cost leaves body unchanged", func(t *testing.T) {
		body := []byte(`{"usage":{}}`)
		if out, ok := injectCostIntoUsage(body, nil); ok || string(out) != string(body) {
			t.Errorf("expected no-op for nil cost")
		}
	})
}

func TestApplyResponseCost_EmitsHeadersAndBody(t *testing.T) {
	r := costTestRouter()
	cost := &responseCost{Total: 0.044, Currency: "USD", PerModel: []modelCostEntry{{Model: "premium", Cost: 0.04}, {Model: "balanced", Cost: 0.004}}}
	fallback := []byte(`{"object":"chat.completion","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)

	response := buildResponseBodyContinueResponse(nil, nil)
	r.applyResponseCost(response, cost, fallback)

	common := commonResponseFor(response)
	if common == nil || common.HeaderMutation == nil {
		t.Fatal("expected header mutation")
	}
	got := map[string]string{}
	for _, h := range common.HeaderMutation.SetHeaders {
		got[h.Header.Key] = string(h.Header.RawValue)
	}
	if got[headers.VSRResponseCost] != "0.044" {
		t.Errorf("cost header = %q, want 0.044", got[headers.VSRResponseCost])
	}
	if got[headers.VSRResponseCostCurrency] != "USD" {
		t.Errorf("currency header = %q", got[headers.VSRResponseCostCurrency])
	}
	if got[headers.VSRResponseCostBreakdown] != "premium=0.04;balanced=0.004" {
		t.Errorf("breakdown header = %q", got[headers.VSRResponseCostBreakdown])
	}

	// content-length must be dropped so Envoy recomputes it for the mutated body.
	foundCL := false
	for _, h := range common.HeaderMutation.RemoveHeaders {
		if h == "content-length" {
			foundCL = true
		}
	}
	if !foundCL {
		t.Error("expected content-length removal")
	}

	body, ok := currentResponseBody(response)
	if !ok {
		t.Fatal("expected body mutation")
	}
	var parsed struct {
		Usage map[string]interface{} `json:"usage"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("unmarshal mutated body: %v", err)
	}
	if parsed.Usage["cost"].(float64) != 0.044 {
		t.Errorf("body usage.cost = %v, want 0.044", parsed.Usage["cost"])
	}
}

func TestApplyResponseCost_NilCostIsNoop(t *testing.T) {
	r := costTestRouter()
	response := buildResponseBodyContinueResponse(nil, nil)
	r.applyResponseCost(response, nil, []byte(`{"usage":{}}`))
	common := commonResponseFor(response)
	if common.HeaderMutation != nil || common.BodyMutation != nil {
		t.Error("nil cost must not touch the response")
	}
}

func TestInjectCostIntoStreamingChunk(t *testing.T) {
	cost := &responseCost{Total: 0.044, Currency: "USD", PerModel: []modelCostEntry{{Model: "premium", Cost: 0.044}}}

	t.Run("rewrites only the usage chunk, keeps framing", func(t *testing.T) {
		chunk := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n" +
			"data: [DONE]\n\n"
		out, ok := injectCostIntoStreamingChunk(chunk, cost)
		if !ok {
			t.Fatal("expected injection")
		}
		// The content delta and [DONE] lines must be untouched.
		if !strings.Contains(out, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n") {
			t.Error("content delta line altered")
		}
		if !strings.HasSuffix(out, "data: [DONE]\n\n") {
			t.Error("[DONE] framing altered")
		}
		// The usage line now carries cost.
		if !strings.Contains(out, "\"cost\":0.044") {
			t.Errorf("usage chunk missing cost: %s", out)
		}
	})

	t.Run("no usage chunk is a no-op", func(t *testing.T) {
		chunk := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"
		out, ok := injectCostIntoStreamingChunk(chunk, cost)
		if ok || out != chunk {
			t.Error("expected no-op when no usage present")
		}
	})
}

func TestInjectStreamingCost_SingleModel(t *testing.T) {
	r := costTestRouter()
	ctx := &RequestContext{RequestModel: "premium", StreamingMetadata: map[string]interface{}{
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(1000),
			"completion_tokens": float64(1000),
			"total_tokens":      float64(2000),
		},
	}}
	chunk := "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":1000,\"total_tokens\":2000}}\n\n"
	out, ok := r.injectStreamingCost(chunk, ctx)
	if !ok {
		t.Fatal("expected cost injected into streaming usage chunk")
	}
	// premium: (1000*10 + 1000*30)/1e6 = 0.04
	if !strings.Contains(out, "\"cost\":0.04") {
		t.Errorf("expected cost 0.04 in chunk: %s", out)
	}
	if ctx.ResponseCost == nil || ctx.ResponseCost.Total != 0.04 {
		t.Errorf("ctx.ResponseCost = %+v, want total 0.04", ctx.ResponseCost)
	}
}

func TestResponseCostHeaderFormatting(t *testing.T) {
	cost := &responseCost{
		Total:    0.004215,
		Currency: "USD",
		PerModel: []modelCostEntry{{Model: "premium", Cost: 0.0031}, {Model: "balanced", Cost: 0.001115}},
	}
	if got := formatCost(cost.Total); got != "0.004215" {
		t.Errorf("formatCost = %q, want 0.004215", got)
	}
	if got := cost.breakdownHeaderValue(); got != "premium=0.0031;balanced=0.001115" {
		t.Errorf("breakdownHeaderValue = %q", got)
	}
}
