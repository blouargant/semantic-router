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

package looper

// workflowProgressPerModelUsage is the per-model companion to
// workflowProgressUsage: it groups the same planner/step/extra responses by
// model so a tool-call interrupt response carries the same cost breakdown a
// completed workflow would.
func workflowProgressPerModelUsage(plannerResp *ModelResponse, results []workflowStepResult, extra ...*ModelResponse) []ModelUsage {
	all := []*ModelResponse{plannerResp}
	for _, result := range results {
		all = append(all, result.responses...)
	}
	all = append(all, extra...)
	return GroupUsageByModel(all...)
}
