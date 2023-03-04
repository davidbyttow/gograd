package grad

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDrawDot(t *testing.T) {
	a := Var(2, "a")
	b := Var(-3, "b")
	c := Var(10, "c")

	// d = a*b + c
	d := a.Mul(b).Add(c)
	d.Label = "d"

	d.Backward()
	v := DrawDot(d)
	fmt.Println("\n", v)

	expected := `digraph G {
	rankdir=LR;
	"_1+"->_1;
	"_2*"->_2;
	_2->"_1+";
	_3->"_2*";
	_4->"_2*";
	_5->"_1+";
	"_1+" [ label="+" ];
	"_2*" [ label="*" ];
	_1 [ label="{ d | data 4.0000 | grad 1.0000 }", shape=record ];
	_2 [ label="{ a*b | data -6.0000 | grad 1.0000 }", shape=record ];
	_3 [ label="{ a | data 2.0000 | grad -3.0000 }", shape=record ];
	_4 [ label="{ b | data -3.0000 | grad 2.0000 }", shape=record ];
	_5 [ label="{ c | data 10.0000 | grad 1.0000 }", shape=record ];

}
`
	require.Equal(t, expected, v)
}

func Test_handlesSameNodeTwice(t *testing.T) {
	a := Var(2, "a")
	b := a.Add(a)
	b.Label = "b"

	b.Backward()
	v := DrawDot(b)
	fmt.Println("\n", v)

	expected := `digraph G {
	rankdir=LR;
	"_1+"->_1;
	_2->"_1+";
	_2->"_1+";
	"_1+" [ label="+" ];
	_1 [ label="{ b | data 4.0000 | grad 1.0000 }", shape=record ];
	_2 [ label="{ a | data 2.0000 | grad 2.0000 }", shape=record ];

}
`
	require.Equal(t, expected, v)
}
