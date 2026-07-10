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
	"fmt"
	"strconv"
	"strings"
)

// modelCostEntry is one model's contribution to a request's consolidated cost.
type modelCostEntry struct {
	Model string
	Cost  float64
}

// responseCost is the consolidated cost of every upstream model call made for a
// single request: one entry on the main routing path, several across a looper
// run that fans out to multiple models. It is returned to the client both as
// x-vsr-response-cost* headers and inside the response body usage object. Only
// models with configured pricing contribute; a request whose models are all
// unpriced yields a nil *responseCost (nothing is emitted).
type responseCost struct {
	Total    float64
	Currency string
	PerModel []modelCostEntry
}

// costModelLeg pairs a model name with the token usage attributed to it, the
// unit the cost builder consumes. The main path passes a single leg; the looper
// path converts its per-model usage breakdown into one leg per model.
type costModelLeg struct {
	model string
	usage responseUsageMetrics
}

// buildResponseCost prices each leg with that model's own configured rate and
// sums the total. It returns nil when the router has no config or when not one
// leg resolves to a priced model, so callers can treat "no pricing" and "zero
// cost" distinctly (a priced model that genuinely costs 0 still returns a
// non-nil report). The currency is taken from the first priced leg; mixed
// currencies are not converted (a misconfiguration the validator surfaces).
func (r *OpenAIRouter) buildResponseCost(legs []costModelLeg) *responseCost {
	if r.Config == nil {
		return nil
	}
	report := &responseCost{}
	priced := false
	for _, leg := range legs {
		pricing, ok := r.Config.GetFullModelPricing(leg.model)
		if !ok {
			continue
		}
		cost := costForResponseUsage(leg.usage, pricing)
		report.Total += cost
		report.PerModel = append(report.PerModel, modelCostEntry{Model: leg.model, Cost: cost})
		if report.Currency == "" {
			report.Currency = pricing.Currency
		}
		priced = true
	}
	if !priced {
		return nil
	}
	return report
}

// Map renders the cost as the block embedded under the body usage object. It
// mirrors the header surface: a flat total + currency for simple readers, plus
// a per-model breakdown for attribution.
func (c *responseCost) Map() map[string]interface{} {
	perModel := make([]map[string]interface{}, 0, len(c.PerModel))
	for _, entry := range c.PerModel {
		perModel = append(perModel, map[string]interface{}{
			"model": entry.Model,
			"cost":  entry.Cost,
		})
	}
	return map[string]interface{}{
		"total":     c.Total,
		"currency":  c.Currency,
		"per_model": perModel,
	}
}

// formatCost renders a cost amount as a plain decimal string for header values,
// trimming trailing zeros so small amounts stay readable (e.g. "0.004215").
func formatCost(amount float64) string {
	return strconv.FormatFloat(amount, 'f', -1, 64)
}

// breakdownHeaderValue renders the per-model split as "model=cost;model=cost" in
// first-called order for the x-vsr-response-cost-breakdown header.
func (c *responseCost) breakdownHeaderValue() string {
	parts := make([]string, 0, len(c.PerModel))
	for _, entry := range c.PerModel {
		parts = append(parts, fmt.Sprintf("%s=%s", entry.Model, formatCost(entry.Cost)))
	}
	return strings.Join(parts, ";")
}

// injectCostIntoUsage adds the cost block to the "usage" object of an
// OpenAI-shaped JSON response body. It is deliberately conservative: if the body
// is not a JSON object, carries no top-level "usage" object, or fails to
// re-marshal, it returns the body unchanged with ok=false so the caller leaves
// the upstream bytes intact. Extra keys inside usage are ignored by
// OpenAI-compatible clients, so this never breaks a strict consumer.
func injectCostIntoUsage(body []byte, cost *responseCost) ([]byte, bool) {
	if cost == nil {
		return body, false
	}
	var root map[string]interface{}
	if err := json.Unmarshal(body, &root); err != nil {
		return body, false
	}
	usage, ok := root["usage"].(map[string]interface{})
	if !ok {
		return body, false
	}
	usage["cost"] = cost.Total
	usage["currency"] = cost.Currency
	usage["cost_per_model"] = cost.Map()["per_model"]
	root["usage"] = usage

	out, err := json.Marshal(root)
	if err != nil {
		return body, false
	}
	return out, true
}
