package grad_test

import (
	"testing"

	"github.com/gograd/grad"
	"github.com/stretchr/testify/require"
)

func TestForward(t *testing.T) {
	a := grad.Var(2, "a")
	b := grad.Var(-3, "b")
	c := grad.Var(10, "c")

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
