package main

import (
	"fmt"

	"github.com/davidbyttow/gograd/grad"
)

func main() {
	x := grad.Scalars(2, 3, -1)
	n := grad.NewMLP(3, []int{4, 4, 1})
	n.Act(x)

	xs := [][]*grad.Scalar{
		grad.Scalars(2, 3, -1),
		grad.Scalars(3, -1, 0.5),
		grad.Scalars(0.5, 1, 1),
		grad.Scalars(1, 1, -1),
	}
	ys := grad.Scalars(1, -1, -1, 1) // targets

	steps := 50

	var loss *grad.Scalar
	var ypred []*grad.Scalar
	for i := 0; i < steps; i++ {

		// forward pass
		ypred = nil
		for _, x := range xs {
			ypred = append(ypred, n.Act(x)...)
		}

		loss = grad.MeanSquaredError(ys, ypred)

		// backward pass
		n.Parameters().ZeroGrad()
		loss.Backward()

		// update
		for _, p := range n.Parameters() {
			p.Descend(0.1)
		}
		fmt.Println(i, loss.Data())
	}

	// predicted values
	for _, yp := range ypred {
		fmt.Println(yp)
	}
}
