package grad

import (
	"fmt"
	"math"
)

// Scalar is a node like Tensor except for scalar values only
// It implements simple backpropagation and gradient computation
type Scalar struct {
	data     float64
	grad     float64
	children []*Scalar
	backward func()
	op       string
	Label    string
}

func Const(data float64) *Scalar {
	return &Scalar{data: data}
}

func Var(data float64, label string) *Scalar {
	return &Scalar{data: data, Label: label}
}

func Res(data float64, children []*Scalar, op string) *Scalar {
	return &Scalar{data: data, op: op, children: children}
}

func (v *Scalar) Data() float64 {
	return v.data
}

func (v *Scalar) Grad() float64 {
	return v.grad
}

func (v *Scalar) Add(other *Scalar) *Scalar {
	out := &Scalar{
		data:     v.data + other.data,
		children: []*Scalar{v, other},
		op:       "+",
		Label:    fmt.Sprintf("%s+%s", v.Label, other.Label),
	}
	out.backward = func() {
		v.grad += 1 * out.grad
		other.grad += 1 * out.grad
	}
	return out
}

func (v *Scalar) Mul(other *Scalar) *Scalar {
	out := &Scalar{
		data:     v.data * other.data,
		children: []*Scalar{v, other},
		op:       "*",
		Label:    fmt.Sprintf("%s*%s", v.Label, other.Label),
	}
	out.backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}
	return out
}

func (v *Scalar) TanH() *Scalar {
	x := v.data
	//t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	t := math.Tanh(x)
	out := &Scalar{
		data:     t,
		children: []*Scalar{v},
		Label:    fmt.Sprintf("tanh(%s)", v.Label),
		op:       "tanh",
	}
	out.backward = func() {
		v.grad += (1 - t*t) * out.grad
	}
	return out
}

func (v *Scalar) Neg() *Scalar {
	return v.Mul(Const(-1))
}

func (v *Scalar) ReLU() *Scalar {
	var f float64
	if v.data > 0 {
		f = v.data
	}
	out := &Scalar{
		data:     f,
		children: []*Scalar{v},
		Label:    fmt.Sprintf("ReLU(%s)", v.Label),
		op:       "ReLU",
	}
	out.backward = func() {
		if out.data > 0 {
			v.grad += out.grad
		}
	}
	return out
}

// Backward calls the backward function for each method in a breadth-first way starting
// with the given root scalar. This iteratively builds the gradient at each depth because
// of the chain rule
func (v *Scalar) Backward() {
	var topo []*Scalar
	var build func(*Scalar)
	visited := map[*Scalar]bool{}

	build = func(cv *Scalar) {
		if _, found := visited[cv]; !found {
			visited[cv] = true
			for _, child := range cv.children {
				build(child)
			}
			topo = append(topo, cv)
		}
	}
	build(v)
	v.grad = 1
	reverseSlice(topo)
	for _, cv := range topo {
		if cv.backward != nil {
			cv.backward()
		}
	}
}

func (v *Scalar) String() string {
	return fmt.Sprintf("Scalar(%s, data=%v, grad=%v)", v.Label, v.data, v.grad)
}

func reverseSlice[T any](x []T) []T {
	for i, j := 0, len(x)-1; i < j; i, j = i+1, j-1 {
		x[i], x[j] = x[j], x[i]
	}
	return x
}
