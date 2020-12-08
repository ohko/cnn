package cnn

import (
	"encoding/json"
	"io/ioutil"
)

// Convolution ...
// output size: Output=((Width-Filter)+2*Padding)/Step+1
// O 是输出尺寸，K 是过滤器尺寸，P 是填充，S 是步幅
type Convolution struct {
	count     int
	step      int
	filters   []*Matrix
	lastInput *Matrix
}

// NewConvolution ...
func NewConvolution(w, h, d, count, step int) *Convolution {
	o := &Convolution{
		count:   count,
		step:    step,
		filters: make([]*Matrix, count),
	}
	for i := 0; i < count; i++ {
		o.filters[i] = NewMatrix(w, h, d, func() float64 {
			return Random(0, 1) - 0.5
		})
	}
	return o
}

// Forward ...
func (o *Convolution) Forward(inputs *Matrix) *Matrix {
	// tmps := make([]*Matrix, len(o.filters))
	// for _, v := range o.filters {
	tmp := inputs.Clone()
	tmp.Padding(1, 1)
	o.lastInput = tmp
	// tmps[k] = tmp.Conv(o.filters)

	// }
	return tmp.Conv(o.filters)
}

// Backprop ...
func (o *Convolution) Backprop(pools *Matrix, learnRate float64) *Matrix {
	last := o.lastInput.Clone()
	// last.Padding(1, 1)
	// print3(true, "last:", last.GetData())
	// print3(true, "pools:", pools.GetData())
	diff := last.Conv([]*Matrix{pools})

	for f := 0; f < len(o.filters); f++ {
		for k := 0; k < diff.D; k++ {
			for y := 0; y < diff.H; y++ {
				for x := 0; x < diff.W; x++ {
					// o.filters.data[k%o.filters.d][y][x] += diff.data[k][y][x]
					o.filters[f].Data[k][y][x] += diff.Data[k][y][x]
				}
			}
		}
	}
	// for _, v := range o.filters {
	// 	print3(true, "o.filters:", v.GetData())
	// }

	// pools.Padding(o.lastInput.w-pools.w, o.lastInput.h-pools.h)
	// print3(true, "pools:", pools.GetData())
	// for _, v := range o.filters {
	// 	print3(true, "o.filters:", v.GetData())
	// }
	r := make([]*Matrix, len(o.filters))
	for k, v := range o.filters {
		_v := v.Clone()
		_v.Rotate180()
		r[k] = _v
	}
	// for _, v := range r {
	// 	print3(true, "r:", v.GetData())
	// }
	pools.Padding(1, 1)
	out := pools.Conv(r)

	// print3(true, "out", out.GetData())
	return out
}

// ToJSON ...
func (o *Convolution) ToJSON() string {
	bs, _ := json.Marshal(o.filters)
	return string(bs)
}

// FromJSON ...
func (o *Convolution) FromJSON(bs string) error {
	return json.Unmarshal([]byte(bs), &o.filters)
}

// SaveJSON ...
func (o *Convolution) SaveJSON(file string) error {
	return ioutil.WriteFile(file, []byte(o.ToJSON()), 0644)
}

// LoadJSON ...
func (o *Convolution) LoadJSON(file string) error {
	bs, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	return o.FromJSON(string(bs))
}
