package neatgo

// Convolution ...
type Convolution struct {
	w, h      int
	step      int
	depth     int
	filters   [][][]float64
	lastInput [][]float64
}

// NewConvolution ...
func NewConvolution(w, h, depth, step int) *Convolution {
	o := &Convolution{w: w, h: h, step: step, depth: depth}
	for i := 0; i < depth; i++ {
		filter := [][]float64{}
		for y := 0; y < h; y++ {
			row := []float64{}
			for x := 0; x < w; x++ {
				row = append(row, Random(0, 1))
			}
			filter = append(filter, row)
		}
		o.filters = append(o.filters, filter)
	}
	return o
}

// Forward ...
func (o *Convolution) Forward(input [][]float64) [][][]float64 {
	o.lastInput = input
	return matrixMul(input, o.filters)
}

// Backprop ...
func (o *Convolution) Backprop(pools [][][]float64, maxIndex [][]int, learnRate float64) [][][]float64 {
	// print2("o.lastInput:", o.lastInput)
	// print3("pools:", pools)
	// print3("o.filters:", o.filters)
	diff := matrixMul(o.lastInput, pools)
	// print3("diff:", diff)

	for k := 0; k < len(diff); k++ {
		for y := 0; y < len(diff[0]); y++ {
			for x := 0; x < len(diff[0][0]); x++ {
				o.filters[k][y][x] += diff[k][y][x]
			}
		}
	}
	// print3("o.filters 2", o.filters)

	return o.filters
}
