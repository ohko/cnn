package cnn

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func padding(input [][]float64, n int) [][]float64 {
	out := make([][]float64, len(input)+n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, len(input[0])+n*2)
	}
	for y := 0; y < len(input); y++ {
		tmp := append([]float64{0}, input[y]...)
		out[n+y] = append(tmp, []float64{0}...)
	}
	for i := 0; i < n; i++ {
		out = append(out, make([]float64, len(input[0])+n*2))
	}

	return out
}

func clone2(input [][]float64) [][]float64 {
	out := make([][]float64, len(input))
	for y := 0; y < len(input); y++ {
		row := make([]float64, len(input[0]))
		for x := 0; x < len(input[0]); x++ {
			row[x] = input[y][x]
		}
		out[y] = row
	}
	return out
}

func print1(debug bool, name string, input []float64) {
	if !debug {
		return
	}
	fmt.Println("===", name, "===")
	fmt.Printf("%.4f\n", input)
}

func print2(debug bool, name string, input [][]float64) {
	if !debug {
		return
	}
	fmt.Println("===", name, "===")
	for y := 0; y < len(input); y++ {
		fmt.Printf("%.4f\n", input[y])
	}
}
func print3(debug bool, name string, input [][][]float64) {
	if !debug {
		return
	}
	fmt.Println("===", name, "===")
	for y := 0; y < len(input); y++ {
		print2(debug, "", input[y])
	}
}

func Normalization(input [][]float64) [][]float64 {
	for y := 0; y < len(input); y++ {
		for x := 0; x < len(input[0]); x++ {
			input[y][x] /= 0xff
		}
	}
	return input
}

var randBool bool

// Random ...
func Random(min, max float64) float64 {
	if min == 0 && max == 0 {
		return 0
	}

	if !randBool {
		randBool = true
		rand.Seed(time.Now().UnixNano())
	}
	return min + rand.Float64()*(max-min)
}
func sigmoid(x float64) float64 {
	return (1 / (1 + math.Exp(-x)))
}

// GetMax ...
func GetMax(data []float64) (int, float64) {
	i, max := 0, 0.0
	for k, v := range data {
		if v > max {
			max = v
			i = k
		}
	}
	return i, max
}

// Uint8ToFloat64 ...
func Uint8ToFloat64(img [][]uint8) [][]float64 {
	out := [][]float64{}
	for y := 0; y < len(img); y++ {
		row := []float64{}
		for x := 0; x < len(img[0]); x++ {
			row = append(row, float64(img[y][x]))
		}
		out = append(out, row)
	}
	return out
}
