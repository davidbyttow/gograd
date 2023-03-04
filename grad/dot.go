package grad

import (
	"fmt"

	"github.com/awalterschulze/gographviz"
)

type edgeT struct {
	src *Scalar
	dst *Scalar
}

type dagT struct {
	ids   map[*Scalar]string
	nodes []*Scalar
	edges []*edgeT
}

func (dag *dagT) build(n *Scalar) {
	var trace func(*Scalar)
	trace = func(n *Scalar) {
		if _, found := dag.ids[n]; !found {
			id := "_" + nextID()
			dag.ids[n] = id
			dag.nodes = append(dag.nodes, n)
			for _, c := range n.children {
				dag.edges = append(dag.edges, &edgeT{c, n})
				trace(c)
			}
		}
	}
	dag.ids = map[*Scalar]string{}
	dag.nodes = nil
	dag.edges = nil
	trace(n)
}

func DrawDot(n *Scalar) string {
	g := gographviz.NewEscape()
	g.SetName("G")
	g.SetDir(true)
	g.Attrs.Add("rankdir", "LR")
	dag := &dagT{}
	dag.build(n)

	for _, n := range dag.nodes {
		id := dag.ids[n]
		var label string
		if n.Label != "" {
			label = fmt.Sprintf(`{ %s | data %0.4f | grad %0.4f }`, n.Label, n.data, n.grad)
		} else {
			label = fmt.Sprintf(`{ data %0.4f | grad %0.4f }`, n.data, n.grad)
		}
		g.AddNode("G", id, map[string]string{
			"label": label,
			"shape": "record",
		})
		if n.op != "" {
			opId := fmt.Sprintf(`"%s%s"`, id, n.op)
			g.AddNode("G", opId, map[string]string{
				"label": n.op,
			})
			g.AddEdge(opId, id, true, nil)
		}
	}
	for _, e := range dag.edges {
		dstId := dag.ids[e.dst]
		opId := fmt.Sprintf(`"%s%s"`, dstId, e.dst.op)
		g.AddEdge(dag.ids[e.src], opId, true, nil)
	}
	return g.String()
}

var nextUniqueID = 1

func nextID() string {
	defer func() { nextUniqueID++ }()
	return fmt.Sprintf("%d", nextUniqueID)
}
