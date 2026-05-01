// Copyright 2026 The gVisor Authors.
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

// Package vllm provides a benchmark for vllm on Kubernetes.
package vllm

import (
	"context"
	_ "embed"
	"fmt"
	"hash/fnv"
	"strings"
	"testing"
	"time"

	"gvisor.dev/gvisor/pkg/sync"
	k8s "gvisor.dev/gvisor/test/kubernetes"
	"gvisor.dev/gvisor/test/kubernetes/benchmarks/profiling"
	"gvisor.dev/gvisor/test/kubernetes/benchmarks/utils"
	"gvisor.dev/gvisor/test/kubernetes/benchmetric"
	"gvisor.dev/gvisor/test/kubernetes/k8sctx"
	"gvisor.dev/gvisor/test/kubernetes/testcluster"
	"gvisor.dev/gvisor/test/tpu/vllm"
	v13 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

type vllmPodServer struct {
	cluster     *testcluster.TestCluster
	clientImage string
	pod         *v13.Pod
	service     *v13.Service
}

func (vps *vllmPodServer) InstrumentedRequest(ctx context.Context, argvFn func(hostPort string) []string) ([]byte, error) {
	return vps.cluster.ExecRequestInClientPod(ctx, vps.service, vps.pod.ObjectMeta.Namespace, vps.clientImage, "vllm-client", argvFn)
}

func (vps *vllmPodServer) Logs(ctx context.Context) (string, error) {
	return vps.cluster.ReadPodLogs(ctx, vps.pod)
}

func atLeastNWords(wantNWords int) func(prompt *vllm.Prompt, response *vllm.FullResponse) (*vllm.Prompt, error) {
	return func(prompt *vllm.Prompt, response *vllm.FullResponse) (*vllm.Prompt, error) {
		if err := utils.CheckAtLeastNWords(response.Text(), wantNWords); err != nil {
			prompt.RaiseTemperature()
			return prompt, err
		}
		return nil, nil
	}
}

func wantSubstring(substring string) func(prompt *vllm.Prompt, response *vllm.FullResponse) (*vllm.Prompt, error) {
	return func(prompt *vllm.Prompt, response *vllm.FullResponse) (*vllm.Prompt, error) {
		if !strings.Contains(strings.ToLower(response.Text()), strings.ToLower(substring)) {
			prompt.RaiseTemperature()
			return prompt, fmt.Errorf("response %q does not contain substring %q", response.Text(), substring)
		}
		return nil, nil
	}
}

type tbLogger struct {
	tb testing.TB
}

func (l *tbLogger) Logf(format string, args ...any) {
	l.tb.Logf(format, args...)
}

func (l *tbLogger) Name() string {
	return l.tb.Name()
}

// BenchmarkVLLM runs vllm benchmarks for a single cluster.
func BenchmarkVLLM(ctx context.Context, t *testing.T, k8sCtx k8sctx.KubernetesContext, cluster *testcluster.TestCluster) {
	benchmarkNS := cluster.Namespace(testcluster.NamespaceBenchmark)
	if err := benchmarkNS.Reset(ctx); err != nil {
		t.Fatalf("cannot reset namespace: %v", err)
	}
	defer benchmarkNS.Cleanup(ctx)
	reqWaitCtx, reqWaitCancel := context.WithTimeout(ctx, 5*time.Minute)
	defer reqWaitCancel()
	if err := benchmarkNS.WaitForResources(reqWaitCtx, testcluster.ContainerResourcesRequest{TPU: true}); err != nil {
		t.Fatalf("failed to wait for resources: %v", err)
	}
	endProfiling, err := profiling.MaybeSetup(ctx, t, k8sCtx, cluster, benchmarkNS)
	if err != nil {
		t.Fatalf("Failed to setup profiling: %v", err)
	}
	defer endProfiling()

	logWithTime := func(t *testing.T, format string, values ...any) {
		t.Logf("[%v] "+format, append([]any{time.Now().Format(time.TimeOnly)}, values...)...)
	}

	testCPUArch, err := cluster.RuntimeTestNodepoolArchitecture(ctx)
	if err != nil {
		t.Fatalf("Failed to get runtime test nodepool architecture: %v", err)
	}

	if testCPUArch != testcluster.CPUArchitectureX86 {
		t.Fatalf("Unsupported CPU architecture: %v", testCPUArch)
	}

	serverImage, err := k8sCtx.ResolveImage(ctx, vllmBenchImage)
	if err != nil {
		t.Fatalf("Failed to resolve image: %v", err)
	}
	vllmPod, err := cluster.ConfigurePodForRuntimeTestNodepool(ctx, newVLLMServerPod(benchmarkNS, serverImage))
	if err != nil {
		t.Fatalf("Failed to configure pod for runtime nodepool: %v", err)
	}
	vllmPod, err = testcluster.SetContainerResources(vllmPod, "", testcluster.ContainerResourcesRequest{TPU: true})
	if err != nil {
		t.Fatalf("Failed to set container resources: %v", err)
	}

	vllmPod, err = cluster.CreatePod(ctx, vllmPod)
	if err != nil {
		t.Fatalf("Failed to create vllm pod: %v", err)
	}
	defer cluster.DeletePod(ctx, vllmPod)
	logWithTime(t, "Waiting for vllm server pod to start, this may take a long time (tens of minutes) if this is the first time the image is being downloaded onto the node.")
	startCtx, startCtxCancel := context.WithTimeout(ctx, 90*time.Minute)
	if err := cluster.WaitForPodRunning(startCtx, vllmPod); err != nil {
		t.Fatalf("Failed to wait for vllm server pod: %v", err)
	}
	startCtxCancel()
	logWithTime(t, "vllm server pod started on Kubernetes but not yet initialized.")
	vllmService := newVLLMService(benchmarkNS)
	vllmService, err = cluster.CreateService(ctx, vllmService)
	if err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}
	defer cluster.DeleteService(ctx, vllmService)
	vllmClientImage, err := k8sCtx.ResolveImage(ctx, vllmBenchClientImage)
	if err != nil {
		t.Fatalf("Failed to resolve image: %v", err)
	}
	vllmServer := &vllmPodServer{
		cluster:     cluster,
		clientImage: vllmClientImage,
		service:     vllmService,
		pod:         vllmPod,
	}
	llm, err := vllm.New(ctx, vllmServer, &tbLogger{t})
	if err != nil {
		t.Fatalf("Failed to create vllm client against server pod: %v", err)
	}
	logWithTime(t, "vllm server ready.")

	type testCase struct {
		name           string
		query          string
		verifyResponse func(*vllm.Prompt, *vllm.FullResponse) (*vllm.Prompt, error)
	}
	testCases := []testCase{
		{
			name: "HelloWorld",
			query: `
				Reply with the words: "Hello World!".
				Do not reply anything else.
			`,
			verifyResponse: wantSubstring("Hello World!"),
		},
		{
			name: "ExtractMeaning",
			query: `
				Consider the following text:

				"""
				We assembled on the vast green lawn outside as the reactors began
				to slowly wind down. The workers were solemn; the activists who had
				fought against the decommissioning seemed crushed. There was
				supposed to be a speech, but the spokeswoman had lost her notes.
				Outside, the protesters cheered.

				My eyes were drawn to the discarded anti-shutdown banners,
				endlessly reciting the facts.
				The statistics on mortality per trillion kWh (lowest of all energy sources).
				The lifespan of a reactor (70 more years, in our case).
				Minimal land footprint.
				Almost zero emissions.
				No intermittency.
				It became a jumble of words, a litany, almost a kind of glossolalia.
				As far as the protesters outside were concerned,
				it might as well be an alien tongue.

				One thing was clear to them, and that was enough:
				the technology inside this compound was deeply, inherently wrong. It was a sin.

				I could not help but think of that moment on August 6th, 1945,
				when the sky erupted above Shima Hospital.
				My imagination could never fully encompass it.
				How do you imagine more than seventy thousand people annihilated
				in an instant? An ancestor of mine was in that hospital; he went
				from being a doctor, a husband, a father, a pacifist stuck
				in a terrible war, to being a pile of bleached bones covered in rubble,
				all in a single second.
				Not by accident, but because of a choice someone made.
				Not because of a reactor, but because of a bomb.

				Just two days earlier, contradicting his campaign promises,
				the prime minister had suggested that the use of
				"tactical" weapons based on this technology would be an
				acceptable risk if the conflict continued.
				Very few seemed to find this particularly shocking or outrageous.

				They were afraid of reactors, but not of bombs.

				The spokeswoman gave up on finding her notes.
				It was starting to rain, and people were walking away.
				She grabbed the microphone.

				"By the time you regret this, it'll be too late," she said.
				"But honestly, I don't know if I care anymore. Maybe you have it coming."

				The spokeswoman sounded so bitter.
				The protesters didn't mean any harm.
				From their perspective, they were doing good.

				Collective action can change the world when it's deliberate
				and based in reason, but it can also become a mental trap,
				or a societal pressure valve.

				People always think they're doing good when they get
				collectively outraged. That doesn't make them right.

				The Flame will not harm you, Son of Man, if you wield it wisely.
				"""

				Summarize what happened in the above text.
				Then answer the following questions:
				What technology is involved?
				What are the protestors clamoring for?
				What does the spokeswoman mean?
				What does "The Flame" symbolize in the text?
			`,
			verifyResponse: atLeastNWords(32),
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			verifyFn := atLeastNWords(1)
			if test.verifyResponse != nil {
				verifyFn = test.verifyResponse
			}
			numAttempts := 0
			verifyFnCount := func(prompt *vllm.Prompt, resp *vllm.FullResponse) (*vllm.Prompt, error) {
				numAttempts++
				return verifyFn(prompt, resp)
			}
			const testTimeout = 25 * time.Minute
			testCtx, testCancel := context.WithTimeout(ctx, testTimeout)
			defer testCancel()

			prompt := vllm.ZeroTemperaturePrompt(test.query)
			resp, err := llm.PromptUntil(testCtx, prompt, verifyFnCount)
			if err != nil {
				t.Fatalf("cannot prompt: %v", err)
			}
			if !resp.Done() {
				t.Fatalf("warm response did not finish: %v", resp)
			}
			logWithTime(t, "Prompting with query:\n%s\n\nResponse:\n%s\n(end of response)", prompt.Text, resp.Text())
			respHash := fnv.New32()
			respHash.Write([]byte(resp.Text()))
			recorder, err := benchmetric.GetRecorder(ctx)
			if err != nil {
				t.Fatalf("Failed to initialize benchmark recorder: %v", err)
			}
			err = recorder.Record(
				ctx,
				fmt.Sprintf("vLLM/%s", test.name),
				benchmetric.BenchmarkDuration(time.Duration(resp.E2ELatency()*float64(time.Second))),
				benchmetric.SpecificDuration(resp.TimeToFirstToken(), "tok-first"),
				benchmetric.SpecificDuration(resp.TimeToLastToken(), "tok-last"),
				benchmetric.Rate(resp.OutputTokensPerSecond(), "tok"),
				benchmetric.Count(uint64(numAttempts), "prompt-attempts"),
				benchmetric.Count(uint64(resp.NumTokens()), "resp-tokens"),
				benchmetric.Checksum(respHash, "resp"),
			)
			if err != nil {
				t.Fatalf("Failed to record benchmark data: %v", err)
			}
		})
	}

	var wg sync.WaitGroup
	wg.Add(1)
	t.Run("", func(t *testing.T) {
		wg.Done()
	})
	wg.Wait()
}

const (
	vllmServerLabelKey   = "app.kubernetes.io/name"
	vllmServerLabelValue = "vllm-server"
	vllmPort             = 8000
	vllmPodName          = "vllm-server"
	vllmServiceName      = "vllm-service"
	vllmBenchImage       = k8s.ImageRepoPrefix + "tpu/vllm_x86_64:latest"
	vllmBenchClientImage = k8s.ImageRepoPrefix + "gpu/ollama/client_x86_64:latest"
)

func newVLLMServerPod(namespace *testcluster.Namespace, image string) *v13.Pod {
	return &v13.Pod{
		TypeMeta: v1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:      vllmPodName,
			Namespace: namespace.Namespace,
			Labels:    map[string]string{vllmServerLabelKey: vllmServerLabelValue},
		},
		Spec: v13.PodSpec{
			Containers: []v13.Container{
				{
					Name:  vllmPodName,
					Image: image,
					Ports: []v13.ContainerPort{
						{
							Name:          vllmServiceName,
							ContainerPort: vllmPort,
						},
					},
				},
			},
			RestartPolicy: v13.RestartPolicyNever,
		},
	}
}

func newVLLMService(namespace *testcluster.Namespace) *v13.Service {
	return namespace.GetService(vllmServiceName, v13.ServiceSpec{
		Selector: map[string]string{vllmServerLabelKey: vllmServerLabelValue},
		Ports: []v13.ServicePort{
			{
				Name:       vllmServiceName,
				Protocol:   v13.ProtocolTCP,
				Port:       vllmPort,
				TargetPort: intstr.FromString(vllmServiceName),
			},
		},
	})
}
