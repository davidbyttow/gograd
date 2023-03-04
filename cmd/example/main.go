package main

import (
	"fmt"
	"math"

	"github.com/davidbyttow/gograd/grad"
)

func main() {
	n := grad.NewMLP(3, []int{4, 4, 1})

	xs := [][]*grad.Scalar{
		grad.Scalars(2, 3, -1),
		grad.Scalars(3, -1, 0.5),
		grad.Scalars(0.5, 1, 1),
		grad.Scalars(1, 1, -1),
	}
	ys := grad.Scalars(1, -1, -1, 1) // targets

	steps := 20

	var loss *grad.Scalar
	for i := 0; i < steps; i++ {
		// forward pass
		var ypred []*grad.Scalar
		for _, x := range xs {
			ypred = append(ypred, n.Act(x)...)
		}
		loss = calcLoss(ys, ypred)

		// backward pass
		n.Parameters().ZeroGrad()
		loss.Backward()

		// update
		for _, p := range n.Parameters() {
			p.Descend(0.05)
		}
		fmt.Println(loss.Data())

		if math.Abs(loss.Data()) < 0.00001 {
			fmt.Println("completed")
			break
		}
	}

	fmt.Println("result =", loss.Data())
}

func calcLoss(actuals []*grad.Scalar, preds []*grad.Scalar) *grad.Scalar {
	if len(actuals) != len(preds) {
		panic("invalid inputs")
	}
	loss := grad.Val(0)
	for i, actual := range actuals {
		pred := preds[i]
		loss = loss.Add(pred.Sub(actual).Pow(2))
	}
	return loss
}
