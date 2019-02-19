/// This is a sketch of an implementation
/// of a matrix algebra library.
extern crate matrixmultiply;
use std::marker::PhantomData;

use std::ops::{
    Index,
    IndexMut,
    Add,
    Mul,
    Neg,
    Range,
    RangeFrom,
    RangeTo,
    RangeFull,
};

use std::convert::From;

use matrixmultiply::sgemm;

trait Ring: Add + Mul + Neg where Self: Sized {
    fn zero() -> Self;
    fn one() -> Self;
}

macro_rules! ring {
    ($T:ty) => {
        impl Ring for $T {
            fn zero() -> $T { 0 as $T }
            fn one() -> $T { 1 as $T }
        }
    }
}

ring!(f32);

trait Data {
    type Elem;
}

trait DataMut: Data
{}

/* immutable types of data */
impl<T> Data for Vec<T> {
    type Elem = T;
}

impl<'a, T> Data for PhantomData<&'a T> {
    type Elem = T;
}

impl<'a, T> Data for PhantomData<&'a mut T> {
    type Elem = T;
}

/* mutable types of data */
impl<T> DataMut for Vec<T>
{}

impl<'a, T> DataMut for PhantomData<&'a mut T>
{}


#[derive(Debug)]
struct Array<A, U>
    where A: Data<Elem=U>
{
    shape: [usize; 2],
    start: *mut U,
    stride: usize,
    data: A,
}

impl<A, U> Array<A, U>
    where A: Data<Elem=U>
{
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn slice<'a, I, J>(&'a self, is: I, js: J) -> Array<PhantomData<&'a U>, U>
        where I: Into<ArrayRange>,
              J: Into<ArrayRange>
    {
        /* turn into actual ranges */
        let is: ArrayRange = is.into();
        let js: ArrayRange = js.into();

        /* substitute the edge of the array if range is unbounded */
        let iend = is.end.unwrap_or(self.shape[0]);
        let jend = js.end.unwrap_or(self.shape[1]);

        /* compute the size of the view */
        let m = iend - is.start;
        let n = jend - js.start;

        /* compute the pointer to the first element */
        let upper_left = unsafe {
            self.start.add(self.stride() * is.start + js.start)
        };

        Array {
            shape: [m, n],
            stride: self.stride,
            start: upper_left,
            data: PhantomData
        }
    }
}

impl<A, U> Array<A, U>
    where A: DataMut<Elem=U>
{
    pub fn slice_mut<'a, I, J>(&'a mut self, is: I, js: J) -> Array<PhantomData<&'a mut U>, U>
        where I: Into<ArrayRange>,
              J: Into<ArrayRange>
    {
        /* turn into actual ranges */
        let is: ArrayRange = is.into();
        let js: ArrayRange = js.into();

        /* substitute the edge of the array if range is unbounded */
        let iend = is.end.unwrap_or(self.shape[0]);
        let jend = js.end.unwrap_or(self.shape[1]);

        /* compute the size of the view */
        let m = iend - is.start;
        let n = jend - js.start;

        /* compute the pointer to the first element */
        let upper_left = unsafe {
            self.start.add(self.stride() * is.start + js.start)
        };

        Array {
            shape: [m, n],
            stride: self.stride,
            start: upper_left,
            data: PhantomData
        }
    }
}

impl<U> Array<Vec<U>, U> {
    unsafe fn uninitialized_memory(shape: [usize; 2]) -> Self {
        let [m, n] = shape;
        let mut data = Vec::with_capacity(m * n);
        data.set_len(m * n);
        Array {
            shape,
            stride: n,
            start: data.as_mut_ptr(),
            data
        }
    }
}


impl<A, U> Index<[usize; 2]> for Array<A, U>
    where A: Data<Elem=U>
{
    type Output = U;
    #[inline]
    fn index(&self, idx: [usize; 2]) -> &U {
        let [i, j] = idx;
        let stride = self.stride();
        unsafe { & *self.start.add(i * stride + j) }
    }
}

impl<A, U> IndexMut<[usize; 2]> for Array<A, U>
    where A: DataMut<Elem=U>
{
    #[inline]
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut U {
        let [i, j] = idx;
        let stride = self.stride();
        unsafe { &mut *self.start.add(i * stride + j) }
    }
}

type Matrix<R> = Array<Vec<R>, R>;

impl<R: Ring> Matrix<R>
    where R: Clone
{
    #[inline]
    pub fn zero(shape: [usize; 2]) -> Self {
        let [m, n] = shape;
        let mut data = vec! [R::zero(); m * n];
        Matrix {
            shape,
            stride: n,
            start: data.as_mut_ptr(),
            data
        }
    }

    #[inline]
    pub fn id(n: usize) -> Matrix<R> {
        Matrix::from_fn([n, n], |i, j| {
            if i == j {
                R::one()
            } else {
                R::zero()
            }
        })
    }

    #[inline]
    pub fn from_fn<F>(shape: [usize; 2], f: F) -> Self
        where F: Fn(usize, usize) -> R
    {
        let [m, n] = shape;
        unsafe {
            let mut ret = Matrix::uninitialized_memory([m, n]);
            for i in 0 .. m {
                for j in 0 .. n {
                    ret[[i, j]] = f(i, j);
                }
            }
            ret
        }
    }
}

impl Mul for &Matrix<f32> {
    type Output = Matrix<f32>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let [m, k] = self.shape;
        let [l, n] = rhs.shape;
        assert!(l == k);

        unsafe {
            let ret = Matrix::<f32>::uninitialized_memory([m, n]);
            /* matrix multiplication */
            sgemm(
                m, k, n,
                1.,
                self.start, k as isize, 1 as isize,
                rhs.start,  n as isize, 1 as isize,
                0.,
                ret.start,  n as isize, 1 as isize,
            );
            ret
        }
    }
}

macro_rules! matrix {
    /* parses m x n matrices */
    ( $( $( $x:expr )* );* ) => ({
        /* expand macro to a 2D-array */
        let data = [ $( [ $( $x ),* ] ),* ];
        // the two-dimensional array makes
        // the compiler verify that the
        // dimension is the same for each row.
        //
        // This does mean that we need to flat-map
        // to get it into the right storage format.

        /* work out dimensions */
        let m = data.len();
        let n = data[0].len();

        let mut data: Vec<_> = data.into_iter()
            .flat_map(|row| row.into_iter())
            .cloned()
            .collect();

        Matrix {
            shape: [m, n],
            stride: n,
            start: data.as_mut_ptr(),
            data
        }
    });
}

#[derive(Debug)]
struct ArrayRange {
    start: usize,
    end: Option<usize>
}

impl From<Range<usize>> for ArrayRange {
    fn from(r: Range<usize>) -> ArrayRange {
        ArrayRange { start: r.start, end: Some(r.end) }
    }
}

impl From<RangeFull> for ArrayRange {
    fn from(_: RangeFull) -> ArrayRange {
        ArrayRange { start: 0, end: None }
    }
}

impl From<RangeFrom<usize>> for ArrayRange {
    fn from(r: RangeFrom<usize>) -> ArrayRange {
        ArrayRange { start: r.start, end: None }
    }
}

impl From<RangeTo<usize>> for ArrayRange {
    fn from(r: RangeTo<usize>) -> ArrayRange {
        ArrayRange { start: 0, end: Some(r.end) }
    }
}

impl From<usize> for ArrayRange {
    fn from(r: usize) -> ArrayRange {
        ArrayRange { start: r, end: Some(r + 1) }
    }
}

fn main() {
    let mut b = matrix! [
        1.  2.  7.  9. ;
        3.  4.  8.  5. ;
        5.  6.  4.  3.
    ];

    let mut d = b.slice(1..3, 1..4);
    d[[1, 1]] = 5.;
    b[[0, 1]] = 1.;

}
