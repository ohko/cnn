package cnn

import (
	"encoding/json"
	"io/ioutil"
)

// Softmax ...
type Softmax struct {
	x, y, d, out int
	Weights      [][]float64
	lastInput    *Matrix
}

// NewSoftmax ...
func NewSoftmax(x, y, d, out int) *Softmax {
	o := &Softmax{x: x, y: y, d: d, out: out}
	n := x * y * d
	for i := 0; i < out; i++ {
		w := make([]float64, n)
		for j := 0; j < n; j++ {
			w[j] = Random(0, 1) - 0.5
		}
		o.Weights = append(o.Weights, w)
	}
	return o
}

// Forward ...
func (o *Softmax) Forward(input *Matrix) []float64 {
	o.lastInput = input.Clone()
	outputs := make([]float64, o.out)

	// print1(true, "outputs", outputs)
	// print3(true, "input", input.GetData())
	// print2(true, "o.Weights", o.Weights)

	for j := range outputs {
		sum := 0.0
		index := 0
		for k := 0; k < input.D; k++ {
			for y := 0; y < input.H; y++ {
				for x := 0; x < input.W; x++ {
					sum += input.Data[k][y][x] * o.Weights[j][index]
					index++
				}
			}
		}
		outputs[j] = sigmoid(sum)
	}
	return outputs
}

// Backprop ...
func (o *Softmax) Backprop(wants, outputs []float64, learnRate float64) *Matrix {
	rdiff := make([]float64, len(outputs))
	rinput := NewMatrix(o.lastInput.W, o.lastInput.H, o.lastInput.D, nil)

	for k := range outputs {
		rdiff[k] = -(outputs[k] - wants[k]) * outputs[k] * (1 - outputs[k])
		// fmt.Printf("\nrdiff: -(%.4f - %.4f) * %.4f * (1 - %.4f)\n", outputs[k], wants[k], outputs[k], outputs[k])
		// fmt.Println("rdiff:", wants[k], outputs[k], rdiff[k])
	}

	// print1(true, "rdiff:", rdiff)
	// print2(true, "o.Weights 1:", o.Weights)
	for k := 0; k < o.lastInput.D; k++ {
		index := 0
		for y := 0; y < o.lastInput.H; y++ {
			for x := 0; x < o.lastInput.W; x++ {
				sum := 0.0
				for k := range outputs {
					// fmt.Printf("sum: (%.4f * %.4f)\n", rdiff[k], o.Weights[k][index])
					sum += rdiff[k] * o.Weights[k][index]
				}
				// fmt.Printf("r: %.4f * %.4f * (1 - %.4f)\n", sum, o.lastInput.Data[k][y][x], o.lastInput.Data[k][y][x])
				rinput.Data[k][y][x] = sum * o.lastInput.Data[k][y][x] * (1 - o.lastInput.Data[k][y][x])
				index++
			}
		}
	}

	// print3(true, "o.lastInput:", o.lastInput.GetData())
	// print3(true, "rinput:", rinput.Data)
	for j := range outputs {
		for k := 0; k < o.lastInput.D; k++ {
			// index := 0
			for y := 0; y < o.y; y++ {
				for x := 0; x < o.x; x++ {
					// fmt.Printf("%d-%d w: %.4f + %.4f * %.4f * %.4f\n", j, (k*o.x*o.y)+y*o.x+x, o.Weights[j][(k*o.x*o.y)+o.y*y+x], o.lastInput.Data[k][y][x], rdiff[j], learnRate)
					o.Weights[j][(k*o.x*o.y)+y*o.x+x] += (o.lastInput.Data[k][y][x] * rdiff[j] * learnRate)
					// if index == 169 {
					// 	fmt.Println(111)
					// }
					// index++
				}
			}
		}
	}
	// print2(true, "o.Weights 2:", o.Weights)

	// fmt.Println("input:\n", outputs)
	// fmt.Println("rinput:\n", rinput)
	return rinput
}

// ToJSON ...
func (o *Softmax) ToJSON() string {
	bs, _ := json.Marshal(o.Weights)
	return string(bs)
}

// FromJSON ...
func (o *Softmax) FromJSON(bs string) error {
	return json.Unmarshal([]byte(bs), &o.Weights)
}

// SaveJSON ...
func (o *Softmax) SaveJSON(file string) error {
	return ioutil.WriteFile(file, []byte(o.ToJSON()), 0644)
}

// LoadJSON ...
func (o *Softmax) LoadJSON(file string) error {
	bs, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	return o.FromJSON(string(bs))
}

/*
// SoftmaxHelper ...
func SoftmaxHelper(inputW, inputH int, conv *Convolution, pool *MaxPool) (int, int) {
	return int(math.Ceil(float64(inputW-(conv.W-conv.step)) / float64(conv.step) / float64(pool.W))),
	int(math.Ceil(float64(inputH-(conv.H-conv.step)) / float64(conv.step) / float64(pool.H)))
}
*/
