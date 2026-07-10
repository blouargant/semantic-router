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
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

// TestCreateLooperResponse_ConsolidatesCost verifies a multi-model looper run
// bills each leg at its own model's rate and returns the consolidated cost on
// both surfaces (headers + body usage object).
func TestCreateLooperResponse_ConsolidatesCost(t *testing.T) {
	r := costTestRouter()
	resp := &looper.Response{
		Body:        []byte(`{"object":"chat.completion","usage":{"prompt_tokens":2000,"completion_tokens":2000,"total_tokens":4000}}`),
		ContentType: "application/json",
		Model:       "premium",
		PerModelUsage: []looper.ModelUsage{
			{Model: "premium", Usage: looper.TokenUsage{PromptTokens: 1000, CompletionTokens: 1000, TotalTokens: 2000}},
			{Model: "balanced", Usage: looper.TokenUsage{PromptTokens: 1000, CompletionTokens: 1000, TotalTokens: 2000}},
		},
	}

	procResp := r.createLooperResponse(resp, &RequestContext{})
	immediate, ok := procResp.Response.(*ext_proc.ProcessingResponse_ImmediateResponse)
	if !ok {
		t.Fatal("expected immediate response")
	}

	got := map[string]string{}
	for _, h := range immediate.ImmediateResponse.Headers.SetHeaders {
		got[h.Header.Key] = string(h.Header.RawValue)
	}
	// premium: (1000*10+1000*30)/1e6 = 0.04; balanced: (1000*1+1000*3)/1e6 = 0.004
	if got[headers.VSRResponseCost] != "0.044" {
		t.Errorf("cost header = %q, want 0.044", got[headers.VSRResponseCost])
	}
	if got[headers.VSRResponseCostBreakdown] != "premium=0.04;balanced=0.004" {
		t.Errorf("breakdown header = %q", got[headers.VSRResponseCostBreakdown])
	}

	var parsed struct {
		Usage map[string]interface{} `json:"usage"`
	}
	if err := json.Unmarshal(immediate.ImmediateResponse.Body, &parsed); err != nil {
		t.Fatalf("unmarshal looper body: %v", err)
	}
	if parsed.Usage["cost"].(float64) != 0.044 {
		t.Errorf("body usage.cost = %v, want 0.044", parsed.Usage["cost"])
	}
}

// TestCreateLooperResponse_NoPricingLeavesBodyIntact verifies that when no
// model is priced, the looper body and headers carry no cost surface.
func TestCreateLooperResponse_NoPricingLeavesBodyIntact(t *testing.T) {
	r := costTestRouter()
	original := `{"object":"chat.completion","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`
	resp := &looper.Response{
		Body:          []byte(original),
		ContentType:   "application/json",
		PerModelUsage: []looper.ModelUsage{{Model: "free", Usage: looper.TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}}},
	}

	procResp := r.createLooperResponse(resp, &RequestContext{})
	immediate := procResp.Response.(*ext_proc.ProcessingResponse_ImmediateResponse)
	if string(immediate.ImmediateResponse.Body) != original {
		t.Errorf("body mutated despite no pricing: %s", immediate.ImmediateResponse.Body)
	}
	for _, h := range immediate.ImmediateResponse.Headers.SetHeaders {
		if h.Header.Key == headers.VSRResponseCost {
			t.Error("cost header emitted despite no pricing")
		}
	}
}
