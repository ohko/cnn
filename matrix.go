package cnn

// Matrix ...
type Matrix struct {
	W    int
	H    int
	D    int
	Data [][][]float64
	init func() float64
}

// NewMatrix ...
func NewMatrix(w, h, d int, init func() float64) *Matrix {
	return &Matrix{W: w, H: h, D: d, init: init, Data: MakeMatrix(w, h, d, init)}
}

// GetWidth ...
func (o *Matrix) GetWidth() int { return o.W }

// GetHeight ...
func (o *Matrix) GetHeight() int { return o.H }

// GetDepth ...
func (o *Matrix) GetDepth() int { return o.D }

// SetData ...
func (o *Matrix) SetData(data [][][]float64) {
	for k := 0; k < o.D; k++ {
		for y := 0; y < o.H; y++ {
			for x := 0; x < o.W; x++ {
				o.Data[k][y][x] = data[k][y][x]
			}
		}
	}
}

// GetData ...
func (o *Matrix) GetData() [][][]float64 {
	out := MakeMatrix(o.W, o.H, o.D, nil)
	for k := 0; k < o.D; k++ {
		for y := 0; y < o.H; y++ {
			for x := 0; x < o.W; x++ {
				out[k][y][x] = o.Data[k][y][x]
			}
		}
	}
	return out
}

// Clone ...
func (o *Matrix) Clone() *Matrix {
	out := NewMatrix(o.W, o.H, o.D, nil)
	for k := 0; k < o.D; k++ {
		for y := 0; y < o.H; y++ {
			for x := 0; x < o.W; x++ {
				out.Data[k][y][x] = o.Data[k][y][x]
			}
		}
	}
	return out
}

// Conv ...
func (o *Matrix) Conv(filters []*Matrix) *Matrix {
	ow := (o.W-filters[0].W)/1 + 1
	// fmt.Println("ow:", ow, o.W, filters[0].W)
	out := NewMatrix(ow, ow, len(filters), nil)
	size := float64(filters[0].W * filters[0].H)

	// print3(true, "filter", filter.GetData())
	// print3(true, "filter", o.GetData())
	// fmt.Println(o.D * filter.d)
	// print3(true, "out", out.GetData())

	// index := 0
	for f := 0; f < len(filters); f++ {
		// tmp := make([][]float64, o.H-filters[f].h+1)
		for y := 0; y < ow; y++ {
			// row := make([]float64, o.W-filters[f].W+1)
			for x := 0; x < ow; x++ {
				sum1 := 0.0
				for k := 0; k < filters[f].D; k++ {
					sum2 := 0.0
					// htm := []string{}
					for b := 0; b < filters[f].H; b++ {
						for a := 0; a < filters[f].W; a++ {
							// fmt.Printf("%.2f*%.2f=%.2f\t", o.Data[k][y+b][x+a], filters[f].Data[k][b][a], o.Data[k][y+b][x+a]*filters[f].Data[k][b][a])
							// fmt.Println(f, k, y+b, x, b, a, o.W, o.W-2)
							sum2 += o.Data[k][y+b][x+a] * filters[f].Data[k][b][a]
							// htm = append(htm, fmt.Sprintf("%.4f*%.4f=%.4f", o.Data[k][y+b][x+a], filter.Data[k][b][a], o.Data[k][y+b][x+a]*filter.Data[k][b][a]))
						}
					}
					// row[x] = sum2 / 0xff
					// row[x] = sum2 / size
					sum2 /= size
					// fmt.Println(sum2)
					// fmt.Println(strings.Join(htm, " + "), sum2)
					// fmt.Println(out[f].Data[k][y][x], sum2)
					// fmt.Println()
					sum1 += sum2
				}
				sum1 /= float64(filters[f].D)
				// fmt.Println(sum1)
				out.Data[f][y][x] = sum1
				// tmp[y] = row
				// index++
			}
			// fmt.Println(k, filter.d, k)
			// out[k] = tmp
		}
	}

	// n := NewMatrix(len(out[0][0]), len(out[0]), len(out), nil)
	// n.SetData(out)
	return out
}

// Padding ...
func (o *Matrix) Padding(w, h int) {
	for k := 0; k < o.D; k++ {
		if w > 0 {
			for y := range o.Data[k] {
				o.Data[k][y] = append(make([]float64, w), o.Data[k][y]...)
				o.Data[k][y] = append(o.Data[k][y], make([]float64, w)...)
			}
		}
	}
	for k := 0; k < o.D; k++ {
		row1 := make([]float64, o.W+w*2)
		tmp1 := [][]float64{row1}
		o.Data[k] = append(tmp1, o.Data[k]...)

		row2 := make([]float64, o.W+w*2)
		tmp2 := [][]float64{row2}
		o.Data[k] = append(o.Data[k], tmp2...)
	}

	o.W += w * 2
	o.H += h * 2
}

// Append ...
func (o *Matrix) Append(w, h int) {
	for k := 0; k < o.D; k++ {
		if w > 0 {
			for y := range o.Data[k] {
				o.Data[k][y] = append(o.Data[k][y], make([]float64, w)...)
			}
		}
		for i := 0; i < h; i++ {
			o.Data[k] = append(o.Data[k], make([]float64, o.W+w))
		}
	}

	o.W += w
	o.H += h
}

// Rotate180 ...
func (o *Matrix) Rotate180() {
	for k, v := range o.Data {
		tmp := make([]float64, o.W*o.H)
		for y := 0; y < o.H; y++ {
			for x := 0; x < o.W; x++ {
				tmp[y*o.H+x] = v[y][x]
			}
		}
		index := len(tmp) - 1
		for y := 0; y < o.H; y++ {
			for x := 0; x < o.W; x++ {
				o.Data[k][y][x] = tmp[index]
				index--
			}
		}
	}
}

// MakeMatrix ...
func MakeMatrix(w, h, d int, init func() float64) [][][]float64 {
	out := make([][][]float64, d)
	for k := 0; k < d; k++ {
		out[k] = make([][]float64, h)
		for y := 0; y < h; y++ {
			out[k][y] = make([]float64, w)
			if init != nil {
				for i := 0; i < w; i++ {
					out[k][y][i] = init()
				}
			}
		}
	}
	return out
}
