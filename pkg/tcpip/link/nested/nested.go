// Copyright 2020 The gVisor Authors.
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

// Package nested provides helpers to implement the pattern of nested
// stack.LinkEndpoints.
package nested

import (
	"gvisor.dev/gvisor/pkg/sync"
	"gvisor.dev/gvisor/pkg/tcpip"
	"gvisor.dev/gvisor/pkg/tcpip/header"
	"gvisor.dev/gvisor/pkg/tcpip/stack"
)

// Endpoint is a wrapper around stack.LinkEndpoint and stack.NetworkDispatcher
// that can be used to implement nesting safely by providing lifecycle
// concurrency guards.
//
// See the tests in this package for example usage.
//
// +stateify savable
type Endpoint struct {
	child    stack.LinkEndpoint
	embedder stack.NetworkDispatcher

	// mu protects dispatcher.
	mu         sync.RWMutex `state:"nosave"`
	dispatcher stack.NetworkDispatcher
}

var _ stack.GSOEndpoint = (*Endpoint)(nil)
var _ stack.LinkEndpoint = (*Endpoint)(nil)
var _ stack.NetworkDispatcher = (*Endpoint)(nil)

// Init initializes a nested.Endpoint that uses embedder as the dispatcher for
// child on Attach.
//
// See the tests in this package for example usage.
func (e *Endpoint) Init(child stack.LinkEndpoint, embedder stack.NetworkDispatcher) {
	e.child = child
	e.embedder = embedder
}

// DeliverNetworkPacket implements stack.NetworkDispatcher.
func (e *Endpoint) DeliverNetworkPacket(protocol tcpip.NetworkProtocolNumber, pkt *stack.PacketBuffer) {
	e.mu.RLock()
	d := e.dispatcher
	e.mu.RUnlock()
	if d != nil {
		d.DeliverNetworkPacket(protocol, pkt)
	}
}

// DeliverLinkPacket implements stack.NetworkDispatcher.
func (e *Endpoint) DeliverLinkPacket(protocol tcpip.NetworkProtocolNumber, pkt *stack.PacketBuffer) {
	e.mu.RLock()
	d := e.dispatcher
	e.mu.RUnlock()
	if d != nil {
		d.DeliverLinkPacket(protocol, pkt)
	}
}

// Attach implements stack.LinkEndpoint.
func (e *Endpoint) Attach(dispatcher stack.NetworkDispatcher) {
	e.mu.Lock()
	e.dispatcher = dispatcher
	e.mu.Unlock()
	// If we're attaching to a valid dispatcher, pass embedder as the dispatcher
	// to our child, otherwise detach the child by giving it a nil dispatcher.
	var pass stack.NetworkDispatcher
	if dispatcher != nil {
		pass = e.embedder
	}
	e.child.Attach(pass)
}

// IsAttached implements stack.LinkEndpoint.
func (e *Endpoint) IsAttached() bool {
	e.mu.RLock()
	isAttached := e.dispatcher != nil
	e.mu.RUnlock()
	return isAttached
}

// MTU implements stack.LinkEndpoint.
func (e *Endpoint) MTU() uint32 {
	return e.child.MTU()
}

// SetMTU implements stack.LinkEndpoint.
func (e *Endpoint) SetMTU(mtu uint32) {
	e.child.SetMTU(mtu)
}

// Capabilities implements stack.LinkEndpoint.
func (e *Endpoint) Capabilities() stack.LinkEndpointCapabilities {
	return e.child.Capabilities()
}

// MaxHeaderLength implements stack.LinkEndpoint.
func (e *Endpoint) MaxHeaderLength() uint16 {
	return e.child.MaxHeaderLength()
}

// LinkAddress implements stack.LinkEndpoint.
func (e *Endpoint) LinkAddress() tcpip.LinkAddress {
	return e.child.LinkAddress()
}

// SetLinkAddress implements stack.LinkEndpoint.SetLinkAddress.
func (e *Endpoint) SetLinkAddress(addr tcpip.LinkAddress) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.child.SetLinkAddress(addr)
}

// WritePackets implements stack.LinkEndpoint.
func (e *Endpoint) WritePackets(pkts stack.PacketBufferList) (int, tcpip.Error) {
	return e.child.WritePackets(pkts)
}

// Wait implements stack.LinkEndpoint.
func (e *Endpoint) Wait() {
	e.child.Wait()
}

// GSOMaxSize implements stack.GSOEndpoint.
func (e *Endpoint) GSOMaxSize() uint32 {
	if e, ok := e.child.(stack.GSOEndpoint); ok {
		return e.GSOMaxSize()
	}
	return 0
}

// SupportedGSO implements stack.GSOEndpoint.
func (e *Endpoint) SupportedGSO() stack.SupportedGSO {
	if e, ok := e.child.(stack.GSOEndpoint); ok {
		return e.SupportedGSO()
	}
	return stack.GSONotSupported
}

// ARPHardwareType implements stack.LinkEndpoint.ARPHardwareType
func (e *Endpoint) ARPHardwareType() header.ARPHardwareType {
	return e.child.ARPHardwareType()
}

// AddHeader implements stack.LinkEndpoint.AddHeader.
func (e *Endpoint) AddHeader(pkt *stack.PacketBuffer) {
	e.child.AddHeader(pkt)
}

// ParseHeader implements stack.LinkEndpoint.ParseHeader.
func (e *Endpoint) ParseHeader(pkt *stack.PacketBuffer) bool {
	return e.child.ParseHeader(pkt)
}

// Close implements stack.LinkEndpoint.
func (e *Endpoint) Close() {
	e.child.Close()
}

// SetOnCloseAction implement stack.LinkEndpoints.
func (e *Endpoint) SetOnCloseAction(action func()) {
	e.child.SetOnCloseAction(action)
}

// Child returns the child endpoint.
func (e *Endpoint) Child() stack.LinkEndpoint {
	return e.child
}
