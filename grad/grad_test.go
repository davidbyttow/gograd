package grad

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestScalar(t *testing.T) {
	a := Var(2, "a")
	b := Var(-3, "b")
	c := Var(10, "c")

	// d = a*b + c
	d := a.Mul(b).Add(c)
	d.Label = "d"

	// forward pass
	require.EqualValues(t, 4, d.Data())

	// back propagation
	d.Backward()

	// derivative of d with respect to d is identity (1)
	// dd / dd = 1
	require.EqualValues(t, 1, d.Grad())

	// slope of c impacts d linearly given addition of (a*b) + c
	// dc / dd = 1
	require.EqualValues(t, 1, c.Grad())

	// a impacts d by -3 (because b=-3).
	// Proof varying a each step:
	// (1 * -3) + 10 = 7
	// (2 * -3) + 10 = 4
	// (3 * -3) + 10 = 1
	// da / dd = -3
	require.EqualValues(t, -3, a.Grad())
}

func TestAllOps(t *testing.T) {
	a := Val(-4)
	require.EqualValues(t, -4, a.Data())
	b := a.ReLU()
	require.EqualValues(t, 0, b.Data())
	c := b.Add(Val(10))
	require.EqualValues(t, 10, c.Data())
	d := c.Tanh()
	require.EqualValues(t, math.Tanh(10), d.Data())
	e := d.Neg()
	require.EqualValues(t, -math.Tanh(10), e.Data())
	f := e.Pow(2)
	require.EqualValues(t, 1, math.Round(f.Data()))
	g := f.Sub(Val(10))
	require.EqualValues(t, -9, math.Round(g.Data()))

	g.Backward()
	require.EqualValues(t, 1.6489228991292286e-08, b.Grad())
}

func TestMeanSquaredError(t *testing.T) {
	ygts := Scalars(1, 1, 1)
	ypreds := Scalars(2, 2, 2)
	mse := MeanSquaredError(ygts, ypreds)
	mse.Backward()

	require.EqualValues(t, 2, ypreds[0].Grad())
	require.EqualValues(t, 3, mse.Data())
}

func TestNeuron(t *testing.T) {
	ru := randUnit
	randUnit = func() float64 { return 0.5 }
	t.Cleanup(func() {
		randUnit = ru
	})
	expected := math.Tanh(1*0.5 + 2*0.5 + 0.5)
	n := NewNeuron(2)
	out := n.Act([]*Scalar{Val(1), Val(2)})
	require.EqualValues(t, expected, out.Data())
}

func TestMLP(t *testing.T) {
	mlp := NewMLP(3, []int{4, 4, 1})
	outs := mlp.Act([]*Scalar{Val(1), Val(2), Val(3)})
	require.Len(t, outs, 1)

	params := mlp.Parameters()
	params.ZeroGrad()

	out := outs[0]
	out.Backward()
	dot := DrawDot(out)
	fmt.Println(dot)

	params.ZeroGrad()
	for _, p := range params {
		fmt.Println(p.String())
	}
}
