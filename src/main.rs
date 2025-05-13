use image::{Rgb, RgbImage};
use num::Zero;
use num::complex::Complex;

const WIDTH: usize = 512;
const HEIGHT: usize = 512;
const FPS: usize = 24;

const MASS: f32 = 1.;

const MAX_POTENTIAL: f32 = 0.5;

fn potential(x: usize, y: usize) -> f32 {
    let x = x as f32 - WIDTH as f32 / 2.;
    let y = y as f32 - HEIGHT as f32 / 2.;
    let r = (x*x + y*y).sqrt();

    if r < 10. {
        MAX_POTENTIAL + 1.
    } else {
        -2. / r
    }
}

fn main() {
    // set up simulation state as 2d array of complex values
    let a: f32 = 0.0001;
    let mut state: Box<[[Complex<f32>; HEIGHT]; WIDTH]> = Box::new([[Complex::zero();HEIGHT];WIDTH]);

    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            state[x][y] = 
                wavepacket(x as f32 - 10., y as f32 - 40., a);
                //+ wavepacket(x as f32 + 50. as f32, y as f32, a)
                //+ wavepacket(x as f32, y as f32 - 43.3, a);
        }
    }

    // ensure wavefunction is normalized
    normalize(&mut state);

    let num_frames = FPS * 30;
    let start = std::time::Instant::now();
    for i in 0..num_frames {
        println!(
            "generating frame {i} of {num_frames} ({}%) - elapsed {:?} est. {:?} remaining", 
            (i * 100) / num_frames, 
            start.elapsed().as_secs(),
            ((start.elapsed().as_secs_f32() / i as f32) * (num_frames - i) as f32) as u32
        );

        framegen(&state, i);

        for _ in 0..200 {
            sim_step(&mut state, 0.005);
            normalize(&mut state);
        }
    }
    println!("finished last frame");
}

fn framegen(state: &Box<[[Complex<f32>; HEIGHT]; WIDTH]>, num: usize) {
    let img = RgbImage::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
        Rgb(complex_visualizer(&state[x as usize][y as usize]))
    });
    img.save(format!("output/img{:0>4}.bmp", num)).unwrap();
}

fn complex_visualizer(c: &Complex<f32>) -> [u8; 3] {
    // probability visualization
    //let v = ((c.norm_sqr() / 10.).tanh() * 256.) as u8;
    //[v,v,v]

    // actual complex value visualization
    let r = c.arg().cos().powi(2);
    let g = c.arg().sin().powi(2);
    let v = (c.norm_sqr() / 10.).tanh() * 255.;

    [(r*v) as u8, (g*v) as u8, 0]
}

// actual physics fr fr

fn sim_step(state: &mut Box<[[Complex<f32>; HEIGHT]; WIDTH]>, dt: f32) {
    let mut substep_state = Box::new([[Complex::ZERO; HEIGHT]; WIDTH]);

    let mut k1: Box<[[Complex<f32>; HEIGHT]; WIDTH]> = Box::new([[Complex::ZERO; HEIGHT]; WIDTH]);
    let mut k2: Box<[[Complex<f32>; HEIGHT]; WIDTH]> = Box::new([[Complex::ZERO; HEIGHT]; WIDTH]);
    let mut k3: Box<[[Complex<f32>; HEIGHT]; WIDTH]> = Box::new([[Complex::ZERO; HEIGHT]; WIDTH]);
    // not needed, integration done in last step with k4 generation
    //let mut k4: [[Complex<f32>; HEIGHT]; WIDTH] = [[Complex::ZERO; HEIGHT]; WIDTH];

    // calculate k1 and prepare for k2 calculation
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            k1[x][y] = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&state, x, y) - potential(x, y) * state[x][y]);
            substep_state[x][y] = state[x][y] + dt * k1[x][y] * 0.5;
        }
    }
    // calculate k2
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            k2[x][y] = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&substep_state, x, y) - potential(x, y) * state[x][y]);
        }
    }
    // prepare for k3 calculation
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            substep_state[x][y] = state[x][y] + dt * k2[x][y] * 0.5;
        }
    }
    // calculate k3
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            k3[x][y] = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&substep_state, x, y) - potential(x, y) * state[x][y]);
        }
    }
    // prepare for k4 calculation
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            substep_state[x][y] = state[x][y] + dt * k3[x][y];
        }
    }
    // calculate k4
    for x in 1..WIDTH-1 {
        for y in 1..HEIGHT-1 {
            if potential(x, y) < MAX_POTENTIAL {
                let k4 = Complex::<f32>::i() * (0.5 / MASS * discrete_laplace_operator(&substep_state, x, y) - potential(x, y) * state[x][y]);
                substep_state[x][y] = state[x][y] + dt / 6. * (k1[x][y] + 2. * k2[x][y] + 2. * k3[x][y] + k4);
            } else {
                substep_state[x][y] = Complex::ZERO;
            }
            
        }
    }

    *state = substep_state
}

fn discrete_laplace_operator(field: &[[Complex<f32>; HEIGHT]; WIDTH], x: usize, y: usize) -> Complex<f32> {
    unsafe {(1./6.) * (
        field.get_unchecked(x.unchecked_sub(1)).get_unchecked(y.unchecked_sub(1))
        + 4. * field.get_unchecked(x.unchecked_sub(1)).get_unchecked(y)
        + field.get_unchecked(x.unchecked_sub(1)).get_unchecked(y.unchecked_add(1))
        + 4. * field.get_unchecked(x).get_unchecked(y.unchecked_sub(1))
        - 20. * field.get_unchecked(x).get_unchecked(y)
        + 4. * field.get_unchecked(x).get_unchecked(y.unchecked_add(1))
        + field.get_unchecked(x.unchecked_add(1)).get_unchecked(y.unchecked_sub(1))
        + 4. * field.get_unchecked(x.unchecked_add(1)).get_unchecked(y)
        + field.get_unchecked(x.unchecked_add(1)).get_unchecked(y.unchecked_add(1))
    )}
}

fn normalize(state: &mut [[Complex<f32>; HEIGHT]; WIDTH]) {
    let v = state.iter().flat_map(|a| a.iter().enumerate()).map(|(i, z)| {
        let n: f32 = z.norm_sqr();
        if potential(i % WIDTH, i / HEIGHT) < MAX_POTENTIAL { n } else { 0.0 }
    }).sum::<f32>();
    state.iter_mut().for_each(|a| a.iter_mut().enumerate().for_each(|(i, z)| {
        if potential(i % WIDTH, i / HEIGHT) < MAX_POTENTIAL {
            *z *= ((WIDTH * HEIGHT) as f32 / v).sqrt()
        } else {
            *z = Complex::ZERO;
        }
    }));
}

fn wavepacket(x: f32, y: f32, a: f32) -> Complex<f32> {
    let mag = f32::exp(-a.powi(2) * ((x - WIDTH as f32 / 2.).powi(2) + (y - HEIGHT as f32/ 2.).powi(2)));
    let phi = f32::atan2(y, x);
    Complex::from_polar(mag, phi)
}