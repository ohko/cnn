package neatgo

import (
	"math"
)

// MaxPool ...
type MaxPool struct {
	w, h       int
	maxIndex   [][]int
	lastInput  [][][]float64
	padx, pady int
}

// NewMaxPool ...
func NewMaxPool(w, h int) *MaxPool {
	return &MaxPool{w: w, h: h}
}

// Forward ...
func (o *MaxPool) Forward(input [][][]float64) [][][]float64 {
	o.maxIndex = o.maxIndex[:0]
	o.lastInput = input
	out := make([][][]float64, len(input))

	for j := 0; j < len(input); j++ {
		padx := len(input[j][0]) % o.w
		pady := len(input[j]) % o.h
		if padx > 0 {
			padx = o.w - padx
			o.padx = padx
		}
		if pady > 0 {
			pady = o.h - pady
			o.pady = pady
		}

		for i := 0; i < pady; i++ {
			input[j] = append(input[j], make([]float64, len(input[j][0])+padx))
		}
		if padx > 0 {
			for k := range input[j] {
				input[j][k] = append(input[j][k], 0)
			}
		}
		tmp := [][]float64{}
		for y := 0; y < len(input[j]); y += o.h {
			row := []float64{}
			for x := 0; x < len(input[j][0]); x += o.w {
				max := math.Inf(-1)
				index := []int{}
				for b := 0; b < o.h; b++ {
					for a := 0; a < o.w; a++ {
						if input[j][y+b][x+a] > max {
							max = input[j][y+b][x+a]
							index = []int{j, y + b, x + a}
						}
					}
				}
				row = append(row, max)
				o.maxIndex = append(o.maxIndex, index)
			}
			tmp = append(tmp, row)
		}

		out[j] = tmp
	}

	return out
}

// Backprop ...
func (o *MaxPool) Backprop(inputs [][][]float64) [][][]float64 {
	// fmt.Println("maxIndex:", o.maxIndex)
	// print3("lastInput:", o.lastInput)
	// print3("inputs:", inputs)

	arr := []float64{}
	for k := range inputs {
		for y := 0; y < len(inputs[0]); y++ {
			for x := 0; x < len(inputs[0][0]); x++ {
				arr = append(arr, inputs[k][y][x])
			}
		}
	}

	out := make([][][]float64, len(o.lastInput))
	for k := range o.lastInput {
		tmp := make([][]float64, len(o.lastInput[0])-o.pady)
		for y := 0; y < len(o.lastInput[0])-o.pady; y++ {
			tmp[y] = make([]float64, len(o.lastInput[0][0])-o.padx)
		}
		out[k] = tmp
	}

	// for k := range o.lastInput {
	// 	for y := 0; y < len(o.lastInput[0])-o.pady; y++ {
	// 		for x := 0; x < len(o.lastInput[0][0])-o.padx; x++ {
	// 			out[k][y][x] = o.lastInput[k][y][x]
	// 		}
	// 	}
	// }

	index := 0
	for _, v := range o.maxIndex {
		out[v[0]][v[1]][v[2]] = arr[index]
		index++
	}

	// print3("out:", out)
	return out
}
