/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use std::convert::From;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Ring:
    Sized + From<u32> + Neg + Add + AddAssign + Sub + SubAssign + Mul + MulAssign {}

pub trait Field: Ring {
    fn invert(&self) -> Self;
}

pub trait Module<C: ?Sized, R: Ring>: Sized + Neg + Add + AddAssign + Sub + SubAssign {
    type IterMut;

    fn new() -> Self;
    fn get(&self, cell: &C) -> Option<&R>;
    fn get_mut(&self, cell: &C) -> Option<&mut R>;
    fn scalar_mul(&mut self, coef: R);
    fn insert(&self, cell: C, coef: R) -> Option<R>;
    fn iter_mut(&mut self) -> Self::IterMut;
}
