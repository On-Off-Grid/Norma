package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// buildAnthropicBody renders a request the way the provider would.
func buildAnthropicBody(t *testing.T, req CompletionRequest) map[string]any {
	t.Helper()
	p := &anthropicProvider{cfg: Config{Model: "claude-opus-5-5"}}
	raw, err := p.buildBody(req, false)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func cacheMarks(v any) int {
	n := 0
	switch x := v.(type) {
	case map[string]any:
		for k, vv := range x {
			if k == "cache_control" {
				if m, ok := vv.(map[string]any); ok && m["type"] == "ephemeral" {
					n++
					continue
				}
			}
			n += cacheMarks(vv)
		}
	case []any:
		for _, vv := range x {
			n += cacheMarks(vv)
		}
	}
	return n
}

// The tool block is static and resent every call, so the last tool carries a
// breakpoint and no other tool does.
func TestAnthropicCachesToolBlock(t *testing.T) {
	body := buildAnthropicBody(t, CompletionRequest{
		Messages: []Message{UserText("hi")},
		Tools: []ToolSchema{
			{Name: "a", Description: "first", InputSchema: map[string]any{"type": "object"}},
			{Name: "b", Description: "second", InputSchema: map[string]any{"type": "object"}},
		},
	})
	tools, _ := body["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools = %v", body["tools"])
	}
	if cacheMarks(tools[0]) != 0 {
		t.Error("a non-final tool carries a cache breakpoint")
	}
	if cacheMarks(tools[1]) != 1 {
		t.Errorf("last tool has %d breakpoints, want 1", cacheMarks(tools[1]))
	}
}

// The breakpoint goes on the stable prefix, never on the final turn: marking
// the newest message would write a cache entry that is never read.
func TestAnthropicCachesStableMessagePrefixOnly(t *testing.T) {
	msgs := []Message{
		UserText("turn one"),
		{Role: RoleAssistant, Content: []ContentBlock{TextBlock("reply one")}},
		UserText("newest turn"),
	}
	body := buildAnthropicBody(t, CompletionRequest{Messages: msgs})
	got, _ := body["messages"].([]any)
	if len(got) != 3 {
		t.Fatalf("messages = %d", len(got))
	}
	if cacheMarks(got[1]) != 1 {
		t.Errorf("stable prefix has %d breakpoints, want 1", cacheMarks(got[1]))
	}
	if cacheMarks(got[2]) != 0 {
		t.Error("the newest turn carries a cache breakpoint; it would never be read")
	}
}

// A single message is entirely the newest turn, so there is no stable prefix.
func TestAnthropicSingleMessageNotCached(t *testing.T) {
	body := buildAnthropicBody(t, CompletionRequest{Messages: []Message{UserText("only")}})
	if n := cacheMarks(body["messages"]); n != 0 {
		t.Errorf("single message got %d breakpoints, want 0", n)
	}
}

// Anthropic allows at most 4 breakpoints; a full request must stay under it.
func TestAnthropicBreakpointBudget(t *testing.T) {
	body := buildAnthropicBody(t, CompletionRequest{
		System:          []string{"static policy", "dynamic tail"},
		DynamicBoundary: 1,
		Messages: []Message{
			UserText("one"),
			{Role: RoleAssistant, Content: []ContentBlock{TextBlock("two")}},
			UserText("three"),
		},
		Tools: []ToolSchema{{Name: "a", Description: "d", InputSchema: map[string]any{"type": "object"}}},
	})
	if n := cacheMarks(body); n > 4 {
		t.Fatalf("%d breakpoints, Anthropic allows 4", n)
	} else if n != 3 {
		t.Errorf("got %d breakpoints, want 3 (system, tools, messages)", n)
	}
}

// markMessagePrefix must not write through a shared history slice.
func TestMarkMessagePrefixDoesNotMutateInput(t *testing.T) {
	msgs := []Message{
		UserText("one"),
		{Role: RoleAssistant, Content: []ContentBlock{TextBlock("two")}},
		UserText("three"),
	}
	_ = markMessagePrefix(msgs)
	for i, m := range msgs {
		for j, b := range m.Content {
			if b.CacheControl != nil {
				t.Fatalf("input mutated at message %d block %d", i, j)
			}
		}
	}
}

// A thinking block's marshaller emits a fixed field set, so a breakpoint there
// would be dropped silently — it must land on a markable block instead.
func TestAnthropicSkipsThinkingBlockForBreakpoint(t *testing.T) {
	msgs := []Message{
		{Role: RoleAssistant, Content: []ContentBlock{
			TextBlock("visible"),
			{Type: BlockThinking, Thinking: "secret", Signature: "sig"},
		}},
		UserText("newest"),
	}
	out := markMessagePrefix(msgs)
	if out[0].Content[1].CacheControl != nil {
		t.Error("breakpoint placed on a thinking block")
	}
	if out[0].Content[0].CacheControl == nil {
		t.Error("breakpoint did not fall back to the preceding markable block")
	}
	raw, err := json.Marshal(out[0].Content[1])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "cache_control") {
		t.Errorf("thinking block serialised a cache_control: %s", raw)
	}
}
