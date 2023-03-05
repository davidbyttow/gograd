package grad

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

var rng *rand.Rand

func init() {
	rng = rand.New(rand.NewSource(time.Now().UnixNano()))
}

// randUnit generate random float between [-1,1]
var randUnit = func() float64 {
	return rng.Float64()*2 - 1
}

// MLP is a multilayer perceptron
type MLP struct {
	layers []*Layer
}

func NewMLP(numIn int, numOuts []int) *MLP {
	sz := append([]int{numIn}, numOuts...)
	layers := make([]*Layer, len(numOuts))
	for i := 0; i < len(numOuts); i++ {
		l := NewLayer(sz[i], sz[i+1])
		if i != len(numOuts)-1 {
			for _, n := range l.neurons {
				n.nonLinear = true
			}
		}
		layers[i] = l
	}
	return &MLP{layers}
}

func (p *MLP) Act(x []*Scalar) []*Scalar {
	for _, l := range p.layers {
		x = l.Act(x)
	}
	return x
}

func (p *MLP) Parameters() Parameters {
	var params []*Scalar
	for _, l := range p.layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

type Layer struct {
	neurons []*Neuron
}

func NewLayer(numIn int, numOut int) *Layer {
	neurons := make([]*Neuron, numOut)
	for i := 0; i < numOut; i++ {
		neurons[i] = NewNeuron(numIn)
	}
	return &Layer{neurons}
}

func (l *Layer) Act(x []*Scalar) []*Scalar {
	outs := make([]*Scalar, len(l.neurons))
	for i := 0; i < len(outs); i++ {
		outs[i] = l.neurons[i].Act(x)
	}
	return outs
}

func (l *Layer) Parameters() Parameters {
	var params []*Scalar
	for _, n := range l.neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

type Neuron struct {
	weights   []*Scalar
	bias      *Scalar
	nonLinear bool
}

func NewNeuron(numIn int) *Neuron {
	if numIn < 1 {
		panic("invalid neuron input size")
	}
	n := &Neuron{
		bias: Val(randUnit()),
	}
	n.weights = make([]*Scalar, numIn)
	for i := 0; i < numIn; i++ {
		n.weights[i] = Val(randUnit())
	}
	return n
}

func (n *Neuron) Act(x []*Scalar) *Scalar {
	if len(x) != len(n.weights) {
		panic("invalid input count")
	}
	// bias + x1*w1 + x2*w2 + ... + xn*wn
	act := n.bias
	for i := 0; i < len(x); i++ {
		act = act.Add(x[i].Mul(n.weights[i]))
	}

	// TODO(d): handle nonLinear case
	// if n.nonLinear {
	// 	return act.ReLU()
	// }
	// return act

	return act.Tanh()
}

func (n *Neuron) Parameters() Parameters {
	return append(n.weights, n.bias)
}

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

func Val(data float64) *Scalar {
	return &Scalar{data: data}
}

func Var(data float64, label string) *Scalar {
	return &Scalar{data: data, Label: label}
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
	}
	out.backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}
	return out
}

func (v *Scalar) Pow(exp float64) *Scalar {
	out := &Scalar{
		data:     math.Pow(v.data, exp),
		children: []*Scalar{v},
		op:       fmt.Sprintf("**%0.4f", exp),
	}
	out.backward = func() {
		v.grad += exp * math.Pow(v.data, (exp-1)) * out.grad
	}
	return out
}

func (v *Scalar) Tanh() *Scalar {
	x := v.data
	t := math.Tanh(x)
	out := &Scalar{
		data:     t,
		children: []*Scalar{v},
		op:       "tanh",
	}
	out.backward = func() {
		v.grad += (1 - t*t) * out.grad
	}
	return out
}

func (v *Scalar) Neg() *Scalar {
	return v.Mul(Val(-1))
}

func (v *Scalar) Sub(other *Scalar) *Scalar {
	return v.Add(other.Neg())
}

func (v *Scalar) ReLU() *Scalar {
	var f float64
	if v.data > 0 {
		f = v.data
	}
	out := &Scalar{
		data:     f,
		children: []*Scalar{v},
		op:       "ReLU",
	}
	out.backward = func() {
		if out.data > 0 {
			v.grad += out.grad
		}
	}
	return out
}

func (v *Scalar) Descend(delta float64) {
	v.data += -delta * v.grad
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
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

func (v *Scalar) String() string {
	return fmt.Sprintf("Scalar(data=%0.4f, grad=%0.4f)", v.data, v.grad)
}

type Parameters []*Scalar

func (p Parameters) ZeroGrad() {
	for _, pm := range p {
		pm.grad = 0
	}
}

func Scalars(vals ...float64) []*Scalar {
	out := make([]*Scalar, len(vals))
	for i, v := range vals {
		out[i] = Val(v)
	}
	return out
}

func MeanSquaredError(ygts []*Scalar, ypreds []*Scalar) *Scalar {
	if len(ygts) != len(ypreds) {
		panic("invalid inputs")
	}
	loss := Val(0)
	for i, ygt := range ygts {
		pred := ypreds[i]
		loss = loss.Add(pred.Sub(ygt).Pow(2))
	}
	return loss
}
