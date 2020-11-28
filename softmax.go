package neatgo

import "math"

// Softmax ...
type Softmax struct {
	x, y, d   int
	weights   [][]float64
	outputs   []float64
	lastInput [][][]float64
}

// NewSoftmax ...
func NewSoftmax(x, y, d, out int) *Softmax {
	o := &Softmax{x: x, y: y, d: d}
	n := x * y * d
	for i := 0; i < out; i++ {
		w := make([]float64, n)
		for j := 0; j < n; j++ {
			w[j] = Random(0, 1)
		}
		o.weights = append(o.weights, w)
	}
	for i := 0; i < out; i++ {
		o.outputs = append(o.outputs, 0)
	}
	return o
}

// Forward ...
func (o *Softmax) Forward(input [][][]float64) []float64 {
	o.lastInput = input

	for k := range o.outputs {
		sum := 0.0
		// htm := []string{}
		for j := 0; j < len(input); j++ {
			for y := 0; y < o.y; y++ {
				for x := 0; x < o.x; x++ {
					sum += input[j][y][x] * o.weights[k][o.y*y+x]
					// htm = append(htm, fmt.Sprintf("[%d]%.2f*%.2f", o.y*y+x, input[j][y][x], o.weights[k][o.y*y+x]))
				}
			}
		}
		// fmt.Println(strings.Join(htm, " + "))
		// fmt.Println("sum:", sum)
		// o.outputs[k] = sigmoid(sum / 338.0)
		o.outputs[k] = sigmoid(sum)
	}
	return o.outputs
}

// Backprop ...
func (o *Softmax) Backprop(outputs []float64, wants []float64, learnRate float64) [][][]float64 {
	rdiff := make([]float64, len(o.outputs))

	rinput := make([][][]float64, len(o.lastInput))
	for j := 0; j < len(o.lastInput); j++ {
		tmp := make([][]float64, len(o.lastInput[j]))
		for y := 0; y < len(o.lastInput[j]); y++ {
			tmp[y] = make([]float64, len(o.lastInput[j][0]))
		}
		rinput[j] = tmp
	}

	for k := range o.outputs {
		rdiff[k] = -(o.outputs[k] - wants[k]) * o.outputs[k] * (1 - o.outputs[k])
		// fmt.Printf("\nrdiff: -(%.4f - %.4f) * %.4f * (1 - %.4f)\n", o.outputs[k], wants[k], o.outputs[k], o.outputs[k])
		// fmt.Println("rdiff:", wants[k], o.outputs[k], rdiff[k])
	}

	// print2("o.weights 1:", o.weights)
	for j := 0; j < len(o.lastInput); j++ {
		for y := 0; y < o.y; y++ {
			for x := 0; x < o.x; x++ {
				sum := 0.0
				for k := range o.outputs {
					// fmt.Printf("sum: (%.4f * %.4f)\n", rdiff[k], o.weights[k][(j*o.x*o.y)+o.y*y+x])
					sum += rdiff[k] * o.weights[k][(j*o.x*o.y)+o.y*y+x]
				}
				// fmt.Printf("r: %.4f * %.4f * (1 - %.4f)\n", sum, o.lastInput[j][y][x], o.lastInput[j][y][x])
				rinput[j][y][x] = sum * o.lastInput[j][y][x] * (1 - o.lastInput[j][y][x])
			}
		}
	}

	// print3("o.lastInput:", o.lastInput)
	// print1("outputs:", outputs)
	for j := 0; j < len(o.lastInput); j++ {
		for y := 0; y < o.y; y++ {
			for x := 0; x < o.x; x++ {
				for k := range o.outputs {
					// fmt.Printf("w: %.4f - %.4f * %.4f * %.4f\n", o.weights[k][(j*o.x*o.y)+o.y*y+x], o.lastInput[j][y][x], rdiff[k], learnRate)
					o.weights[k][(j*o.x*o.y)+o.y*y+x] += (o.lastInput[j][y][x] * rdiff[k] * learnRate)
				}
			}
		}
	}
	// print2("o.weights 2:", o.weights)

	// fmt.Println("input:\n", outputs)
	// fmt.Println("rinput:\n", rinput)
	return rinput
}

// SoftmaxHelper ...
func SoftmaxHelper(inputW, inputH int, conv *Convolution, pool *MaxPool) (int, int) {
	return int(math.Ceil(float64(inputW-(conv.w-conv.step)) / float64(conv.step) / float64(pool.w))),
		int(math.Ceil(float64(inputH-(conv.h-conv.step)) / float64(conv.step) / float64(pool.h)))
}
