package cnn

import (
	"math"
)

// MaxPool ...
type MaxPool struct {
	W, H       int
	maxIndex   [][]int
	lastInput  *Matrix
	padx, pady int
}

// NewMaxPool ...
func NewMaxPool(w, h int) *MaxPool {
	return &MaxPool{W: w, H: h}
}

// Forward ...
func (o *MaxPool) Forward(input *Matrix) *Matrix {
	o.maxIndex = o.maxIndex[:0]
	o.lastInput = input.Clone()

	padx := input.W % o.W
	pady := input.H % o.H
	if padx > 0 || pady > 0 {
		input.Append(padx, pady)
	}

	out := NewMatrix(input.W/o.W, input.H/o.H, input.D, nil)

	for k := 0; k < input.D; k++ {
		for y := 0; y < input.H; y += o.H {
			for x := 0; x < input.W; x += o.W {
				max := math.Inf(-1)
				index := []int{}
				for b := 0; b < o.H; b++ {
					for a := 0; a < o.W; a++ {
						if input.Data[k][y+b][x+a] > max || max == math.Inf(-1) {
							max = input.Data[k][y+b][x+a]
							index = []int{k, y + b, x + a}
						}
					}
				}
				out.Data[k][y/o.H][x/o.W] = max
				o.maxIndex = append(o.maxIndex, index)
			}
		}
	}

	return out
}

// Backprop ...
func (o *MaxPool) Backprop(inputs *Matrix) *Matrix {
	// fmt.Println("maxIndex:", o.maxIndex)
	// print3(true, "lastInput:", o.lastInput.GetData())
	// print3(true, "inputs:", inputs.GetData())

	arr := []float64{}
	for k := 0; k < inputs.D; k++ {
		for y := 0; y < inputs.H; y++ {
			for x := 0; x < inputs.W; x++ {
				arr = append(arr, inputs.Data[k][y][x])
			}
		}
	}

	out := NewMatrix(o.lastInput.W, o.lastInput.H, o.lastInput.D, nil)

	// for k := range o.lastInput {
	// 	for y := 0; y < len(o.lastInput[0])-o.pady; y++ {
	// 		for x := 0; x < len(o.lastInput[0][0])-o.padx; x++ {
	// 			out[k][y][x] = o.lastInput[k][y][x]
	// 		}
	// 	}
	// }

	index := 0
	for _, v := range o.maxIndex {
		out.Data[v[0]][v[1]][v[2]] = arr[index]
		index++
	}

	// print3(true, "out:", out.GetData())
	return out
}
