import numpy as np

from pyGF2.gf2_add import gf2_add
from pyGF2.gf2_mul import gf2_mul
from pyGF2.gf2_div import gf2_div
from pyGF2.generic_functions import monomial, cyclic_shift
import pyGF2.generic_functions as gf2_help_func

import random


class PolynomialGF2:
    __slots__ = ('array', 'degree')

    def __init__(self, array: 'list or np.ndarray'):
        self.array = np.array(array) % 2
        self.degree = gf2_help_func.degree(self.array)

    def __eq__(self, other):
        return np.array_equal(self.array, other.array)

    def __ne__(self, other):
        return not np.array_equal(self.array, other.array)

    def __add__(self, other):
        return PolynomialGF2(gf2_add(self.array, other.array))

    def __sub__(self, other):
        return PolynomialGF2(gf2_add(self.array, other.array))

    def __mod__(self, other):
        _, remainder = gf2_div(self.array, other.array)
        return PolynomialGF2(remainder)

    def __floordiv__(self, other):
        quotient, _ = gf2_div(self.array, other.array)
        return PolynomialGF2(quotient)

    def __mul__(self, other):
        return PolynomialGF2(gf2_mul(self.array, other.array))

    def __lshift__(self, other):
        # cyclic lshift
        shift_val = int(other)
        if shift_val >= 0:
            return PolynomialGF2(cyclic_shift(self.array, shift_val))
        else:
            raise(ValueError)

    def __rshift__(self, other):
        # cyclic rshift
        shift_val = int(other)
        if shift_val >= 0:
            return PolynomialGF2(cyclic_shift(self.array, -shift_val))
        else:
            raise(ValueError)

    def copy(self):
        return PolynomialGF2(self.array)

    def padding(self, length):
        return PolynomialGF2(gf2_help_func.padding(self.array, length))

    def __repr__(self):
        return str(gf2_help_func.strip_zeros(self.array))

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __missing__(self, key):
        raise(ValueError)


class FireCode(object):
    def __init__(self,
                 correct_len: int,
                 detect_len: int,
                 is_example: bool=False):

        self.is_example: bool = is_example
        self.correct_len: int = int(correct_len)
        self.detect_len: int = int(detect_len)

        self.polynomials_degrees_table: dict = self.__init_table()

        self.word_len: int = None
        self.control_symbol: int = None
        self.information_symbol: int = None
        self.form_polinomial: PolynomialGF2 = None
        self.Cx: PolynomialGF2 = None
        self.Gx: PolynomialGF2 = None

        self.__init_code()  # init values above

    def __repr__(self):
        return f"FireCode({self.word_len}, {self.information_symbol - 1})" + \
               f"\n\nG(x) = {self.Gx}\nC(x) = {self.Cx}\nform polynomial = {self.form_polinomial}\n"

    def __init_table(self) -> dict:
        degrees_dict = {}

        # irreducible polynomials in GF(2)
        degrees_dict[1] = PolynomialGF2([1, 1])
        degrees_dict[2] = PolynomialGF2([1, 1, 1])
        degrees_dict[3] = PolynomialGF2([1, 1, 0, 1])
        degrees_dict[4] = PolynomialGF2([1, 1, 0, 0, 1])
        degrees_dict[5] = PolynomialGF2([1, 0, 1, 0, 0, 1])
        degrees_dict[6] = PolynomialGF2([1, 1, 0, 0, 0, 0, 1])
        degrees_dict[7] = PolynomialGF2([1, 1, 0, 0, 0, 0, 0, 1])
        degrees_dict[8] = PolynomialGF2([1, 0, 1, 1, 1, 0, 0, 0, 1])
        degrees_dict[9] = PolynomialGF2([1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        degrees_dict[10] = PolynomialGF2([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

        return degrees_dict

    def __init_code(self):
        deg_one = self.correct_len  # t <= bs
        deg_two = self.correct_len + self.detect_len - 1  # c >= bs + br - 1
        e = 2 * deg_one - 1

        while (deg_two % e == 0) and deg_two != 0:
            deg_two += 1

        self.word_len = np.lcm(deg_two, e)
        self.control_symbol = deg_one + deg_two

        # k = n - m
        self.information_symbol = self.word_len - self.control_symbol

        self.Cx = PolynomialGF2([0 for _ in range(deg_two + 1)])
        self.Cx[0] = self.Cx[-1] = 1

        self.Gx = self.polynomials_degrees_table[deg_one - 1]

        self.form_polinomial = self.Cx * self.Gx

    def __get_msg(self, corrected_poly: PolynomialGF2, shift) -> PolynomialGF2:
        # move to trash error-correcting part of code
        corrected_poly = \
            PolynomialGF2(corrected_poly.array[(self.control_symbol - 1):])

        divisor = PolynomialGF2(monomial(self.control_symbol + 1))

        informational_poly = corrected_poly % divisor

        return informational_poly

    def encode(self, informational_poly: PolynomialGF2) -> PolynomialGF2:

        code_combination = \
            informational_poly * PolynomialGF2(monomial(self.control_symbol))

        remainder = code_combination % self.form_polinomial

        code_combination = code_combination + remainder

        return code_combination

    def decode(self, encoded_poly: PolynomialGF2) -> PolynomialGF2:
        Gx_remainder = encoded_poly % self.Gx
        Cx_remainder = encoded_poly % self.Cx

        if Gx_remainder.degree == Cx_remainder.degree == 0 \
                and Gx_remainder.array.size == Cx_remainder.array.size == 0:

            print("remainders are equal!")

            if self.is_example:
                print("  decoded:", encoded_poly)

            # if there are no errors
            return self.__get_msg(encoded_poly, 0)

        polinomial_module = \
            PolynomialGF2([0 for _ in range(self.information_symbol)])

        polinomial_module[0] = polinomial_module[-1] = 1

        copy_msg = encoded_poly.copy().padding(self.information_symbol - 1)

        shift_counter = 0

        while Gx_remainder != Cx_remainder:
            copy_msg = copy_msg >> 1
            Gx_remainder = copy_msg % self.Gx
            Cx_remainder = copy_msg % self.Cx
            shift_counter += 1
            print(f"shift: {shift_counter}; rem1 = {Cx_remainder}; rem2 = {Gx_remainder}")

        if False:
            # another variant; shifting using mul on poly(X^1)
            shift = PolynomialGF2(monomial(2))  # [0. 1.] == poly(X^1)
            copy_msg = encoded_poly.copy()
            shift_counter = 0
            while Gx_remainder != Cx_remainder:
                copy_msg = (copy_msg * shift) % polinomial_module
                Gx_remainder = copy_msg % self.Gx
                Cx_remainder = copy_msg % self.Cx
                shift_counter += 1
                print(f"shift: {shift_counter}; rem1 = {Cx_remainder}; rem2 = {Gx_remainder}")
                # print(f"shift={shift_counter};\nCx_remainder={Cx_remainder}; Gx_remainder={Gx_remainder}\n\n")

        # correcting_poly matches error package
        correcting_poly = Gx_remainder.padding(len(polinomial_module))

        if shift_counter != 0:
            shift_counter += 1

        correcting_poly = correcting_poly << shift_counter

        if self.is_example:
            print("   errors:", correcting_poly)

        corrected_poly = encoded_poly + correcting_poly  # everything for this

        if self.is_example:
            print("  decoded:", corrected_poly)

        decoded_msg = self.__get_msg(corrected_poly, shift_counter)

        return decoded_msg


if __name__ == '__main__':

    # random.seed(0)

    fire_code = FireCode(correct_len=3, detect_len=2, is_example=True)

    print(fire_code)

    input_msg = input("format: 101000\nEnter msg: ")

    input_msg = list(map(int, list(input_msg)))

    if len(input_msg) > fire_code.information_symbol:
        print("Too long msg")

    input_msg = PolynomialGF2(input_msg)

    print("    input:", input_msg)

    encoded = fire_code.encode(input_msg)
    print("  encoded:", encoded)

    noise = PolynomialGF2([1, 1])

    noise = noise.padding(len(encoded)) >> 3# random.randint(0, 20)
    print("    noise:", noise)

    noisy_msg = encoded + noise
    print("noisy msg:", noisy_msg)

    decoded = fire_code.decode(noisy_msg)
    print("   output:", decoded)
