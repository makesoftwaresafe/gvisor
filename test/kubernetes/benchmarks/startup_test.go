// Copyright 2024 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package startup

import (
	"context"
	"testing"

	"gvisor.dev/gvisor/test/kubernetes/k8sctx"
	"gvisor.dev/gvisor/test/kubernetes/k8sctx/kubectlctx"
	"gvisor.dev/gvisor/test/kubernetes/testcluster"
)

func TestStartup(t *testing.T) {
	ctx := context.Background()
	k8sCtx, err := kubectlctx.New(ctx)
	if err != nil {
		t.Fatalf("Failed to get kubernetes context: %v", err)
	}
	k8sctx.ForEachCluster(ctx, t, k8sCtx, func(cluster *testcluster.TestCluster) {
		t.Run(benchName, func(t *testing.T) {
			cluster := cluster
			t.Parallel()
			MeasureStartup(ctx, t, k8sCtx, cluster)
		})
	})
}
