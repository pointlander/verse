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
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tf32"
)

const (
	// Size is the size of the verse
	Size = 8
	// Width is the width of the neural network
	Width = Size * Size
	// Scale is the scale of the verse
	Scale = 100
)

func simulate(n int) {
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

	alpha, eta, iterations := float32(.3), float32(.3), 30
	points, images := make(plotter.XYs, 0, iterations), &gif.GIF{}
	var palette = []color.Color{
		color.RGBA{0, 0, 0, 0xff},
		color.RGBA{0xff, 0xff, 0xff, 0xff},
		color.RGBA{0, 0, 0xff, 0xff},
	}
	for step := 0; step < 256; step++ {
		i := 0
		for i < iterations {
			total := float32(0.0)
			start := time.Now()
			set.Zero()

			total += tf32.Gradient(cost).X[0]
			norm := float32(0)
			for _, p := range set.Weights {
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
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

			points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
			fmt.Println(i, total, time.Now().Sub(start))
			i++
		}

		particles := set.Weights[2]
		verse := image.NewPaletted(image.Rect(0, 0, Size*Scale, Size*Scale), palette)
		for i := 0; i < len(particles.X); i += Width {
			maxX, maxY, max := 0, 0, float32(0.0)
			for y := 0; y < 8; y++ {
				offset := i + 8*y
				for x := 0; x < 8; x++ {
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

			for x := 0; x < int(float64(step)*Size*Scale/256.0); x++ {
				for y := Size*Scale - 10; y < Size*Scale; y++ {
					verse.Set(x, y, color.RGBA{0, 0, 0xff, 0xff})
				}
			}
		}
		images.Image = append(images.Image, verse)
		images.Delay = append(images.Delay, 100)

		for i := range set.Weights {
			w := set.Weights[i]
			for j := range w.X {
				w.X[j] += float32(rand.NormFloat64() * .001)
			}
		}
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

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs_%d.png", n))
	if err != nil {
		panic(err)
	}

	out, err := os.Create(fmt.Sprintf("verse_%d.gif", n))
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

	simulate(1)
	simulate(2)
	simulate(3)
	simulate(8)
}
