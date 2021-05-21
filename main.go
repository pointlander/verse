// Copyright 2021 The Verse Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
)

const (
	// Size is the size of the verse
	Size = 32
	// Width is the width of the neural network
	Width = Size * Size
	// Scale is the scale of the verse
	Scale = 25
	// QuantumWidth is the width of the quantum verse
	QuantumWidth = 256
)

var (
	// VerseMode
	verseMode = flag.Bool("verse", false, "verse mode")
	// MultiVerseMode
	multiVerseMode = flag.Bool("multi", false, "multi verse mode")
)

func multiVerse(seed int64, factor float64, width int, last *tc128.Set) (s *tc128.Set, state uint64) {
	rand.Seed(seed)

	set := tc128.NewSet()
	set.Add("aw1", width+1, width+1)
	set.Add("ab1", width+1)
	set.Add("states", width+1)

	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
	}

	for i := range set.Weights {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		} else {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		}
	}

	if last != nil {
		for i := 0; i < width; i++ {
			for j := 0; j < width; j++ {
				set.Weights[0].X[j*(width+1)+i] = last.Weights[0].X[j*width+i]
			}
			set.Weights[1].X[i] = last.Weights[1].X[i]
			set.Weights[2].X[i] = last.Weights[2].X[i]
		}
	}

	l1 := tc128.Softmax(tc128.Add(tc128.Mul(set.Get("aw1"), set.Get("states")), set.Get("ab1")))
	cost := tc128.Quadratic(set.Get("states"), l1)

	eta, iterations := complex128(.3), 512
	i := 0
	for i < iterations {
		set.Zero()

		tc128.Gradient(cost)
		sum := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm := float64(math.Sqrt(float64(sum)))
		scaling := float64(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for _, p := range set.Weights {
			for l, d := range p.D {
				p.X[l] -= eta * d * complex(scaling, 0)
			}
		}

		x := set.Weights[0].X
		x[rand.Intn(len(x))] = complex(norm, 0)

		max := 0.0
		for i, j := range set.Weights[2].X {
			if mag := cmplx.Abs(j); mag > max {
				state, max = uint64(i), mag
			}
		}

		i++
	}

	return &set, state
}

func verse(factor float64) {
	rand.Seed(1)

	set := tc128.NewSet()
	set.Add("aw1", QuantumWidth, QuantumWidth)
	set.Add("ab1", QuantumWidth)
	set.Add("states", QuantumWidth)

	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
	}

	for i := range set.Weights {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		} else {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		}
	}

	l1 := tc128.Softmax(tc128.Add(tc128.Mul(set.Get("aw1"), set.Get("states")), set.Get("ab1")))
	cost := tc128.Quadratic(set.Get("states"), l1)

	eta, iterations := complex128(.3), 512
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := complex128(0)
		start := time.Now()
		set.Zero()

		total += tc128.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm := float64(math.Sqrt(float64(sum)))
		scaling := float64(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for _, p := range set.Weights {
			for l, d := range p.D {
				p.X[l] -= eta * d * complex(scaling, 0)
			}
		}

		x := set.Weights[0].X
		x[rand.Intn(len(x))] = complex(norm, 0)

		state, max := uint64(0), 0.0
		for i, j := range set.Weights[2].X {
			if mag := cmplx.Abs(j); mag > max {
				state, max = uint64(i), mag
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: cmplx.Abs(total)})
		fmt.Println(i, cmplx.Abs(total), time.Now().Sub(start), state, max)
		i++
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "verse.png")
	if err != nil {
		panic(err)
	}
}

func simulate(name string, n int, factor float32) {
	rand.Seed(1)

	set := tf32.NewSet()
	set.Add("aw1", Width, Width)
	set.Add("ab1", Width)
	set.Add("particles", Width, n)

	for i := range set.Weights {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, 0)
			}
		} else {
			factor := float32(math.Sqrt(2 / float64(w.S[0])))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rand.NormFloat64())*factor)
			}
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	l1 := tf32.Softmax(tf32.Add(tf32.Mul(set.Get("aw1"), set.Get("particles")), set.Get("ab1")))
	cost := tf32.Avg(tf32.Quadratic(set.Get("particles"), l1))

	alpha, eta, iterations := float32(.3), float32(.3), 2048
	points, images := make(plotter.XYs, 0, iterations), &gif.GIF{}
	var palette = []color.Color{
		color.RGBA{0, 0, 0, 0xff},
		color.RGBA{0xff, 0xff, 0xff, 0xff},
		color.RGBA{0, 0, 0xff, 0xff},
	}
	i := 0
	for i < iterations {
		total := float32(0.0)
		start := time.Now()
		set.Zero()

		total += tf32.Gradient(cost).X[0]
		sum := float32(0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		sum = 0
		for _, p := range set.Weights {
			for j, d := range p.D {
				d += float32(rand.NormFloat64()) * norm * factor
				sum += d * d
				p.D[j] = d
			}
		}
		norm = float32(math.Sqrt(float64(sum)))
		scaling := float32(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for k, p := range set.Weights {
			for l, d := range p.D {
				deltas[k][l] = alpha*deltas[k][l] - eta*d*scaling
				p.X[l] += deltas[k][l]
			}
		}

		particles := set.Weights[2]
		verse := image.NewPaletted(image.Rect(0, 0, Size*Scale, Size*Scale), palette)
		for i := 0; i < len(particles.X); i += Width {
			maxX, maxY, max := 0, 0, float32(0.0)
			for y := 0; y < Size; y++ {
				offset := i + Size*y
				for x := 0; x < Size; x++ {
					if a := particles.X[offset+x]; a > max {
						maxX, maxY, max = x, y, a
					}
				}
			}
			maxX *= Scale
			maxY *= Scale
			for x := 0; x < Scale; x++ {
				for y := 0; y < Scale; y++ {
					var dx, dy float32 = Scale/2 - float32(x), Scale/2 - float32(y)
					d := 2 * float32(math.Sqrt(float64(dx*dx+dy*dy))) / Scale
					if d < 1 {
						verse.Set(maxX+x, maxY+y, color.RGBA{0xff, 0xff, 0xff, 0xff})
					}
				}
			}
		}

		for x := 0; x < int(float64(i)*Size*Scale/float64(iterations)); x++ {
			for y := Size*Scale - 10; y < Size*Scale; y++ {
				verse.Set(x, y, color.RGBA{0, 0, 0xff, 0xff})
			}
		}

		images.Image = append(images.Image, verse)
		images.Delay = append(images.Delay, 10)
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total, time.Now().Sub(start))
		i++
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%s_epochs_%d.png", name, n))
	if err != nil {
		panic(err)
	}

	out, err := os.Create(fmt.Sprintf("%s_%d.gif", name, n))
	if err != nil {
		panic(err)
	}
	defer out.Close()
	err = gif.EncodeAll(out, images)
	if err != nil {
		panic(err)
	}
}

func main() {
	flag.Parse()

	if *multiVerseMode {
		set, state := multiVerse(1, .1, 2, nil)
		fmt.Println(2, state)
		for i := 2; i < 128; i++ {
			set, state = multiVerse(int64(i), .1, i, set)
			fmt.Println(i, state)
		}
	} else if *verseMode {
		verse(.1)
	} else {
		simulate("verse", 1, 1)
		simulate("verse", 2, 1)
		simulate("verse", 3, 1)
		simulate("verse", 8, 1)
		simulate("verse_blackhole", 8, .1)
	}
}
