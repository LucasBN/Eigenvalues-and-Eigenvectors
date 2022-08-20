import numpy as np

class DatasetChecker:
    def __init__(self, data):
        self.data_type = type(data)
        self.data_type_checker(self.data_type)
        self.check_shape(data)

    def data_type_checker(self, data_type):
        """
        Raise error if not data type is not 'class <list>'
        :param data_type:
        """
        if data_type != list:
            raise Exception(f"Can not create dataset from {data_type} object. You have to pass list.")

    def check_shape(self, data):
        """
        Raise an error if not data is square matrix
        :param data:
        """
        row_length = len(data)
        col_length = len(data[0])
        if row_length != col_length:
            raise Exception("Length of all rows and columns must be the same! You have to pass square matrix.")

    def __str__(self):
        return f"\nChecks the data whether is not a square matrix or not a list"


class EigenValuesVectors(DatasetChecker):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.pretty_print_results()

    def get_dimensions(self, matrix):
        """
        Return the dimensions of any given matrix
        :param matrix: A list of lists of `float` values.
        :return: A list of `float` values describing the dimensions of the matrix
        """
        return [len(matrix), len(matrix[0])]

    def find_determinant(self, matrix, excluded=1):
        """
        Return the value of the determinant of any given matrix
        :param matrix: A list of lists of `float` values.
        :param excluded: A `float` value which refers to the value at the
                         row and column along which elements were crossed out.
        :return: A `float` value which is the determinant of the matrix.
        """
        dimensions = self.get_dimensions(matrix)
        if dimensions == [2, 2]:
            return excluded * ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
        else:
            new_matrices = []
            excluded = []
            exclude_row = 0
            for exclude_column in range(dimensions[1]):
                tmp = []
                excluded.append(matrix[exclude_row][exclude_column])
                for row in range(1, dimensions[0]):
                    tmp_row = []
                    for column in range(dimensions[1]):
                        if (row != exclude_row) and (column != exclude_column):
                            tmp_row.append(matrix[row][column])
                    tmp.append(tmp_row)
                new_matrices.append(tmp)
            determinants = [self.find_determinant(new_matrices[j], excluded[j]) for j in range(len(new_matrices))]
            determinant = 0
            for i in range(len(determinants)):
                determinant += ((-1) ** i) * determinants[i]
            return

    def list_multiply(self, list1, list2):
        """
        Return the multiplication of two lists by treating each list as a factor.
        For example, to multiply two lists of length two, use the FOIL method.
        :param list1: A list of `float` values.
        :param list2: A list of `float` values.
        :return: A list of `float` values containing the result of the
                 multiplication.
        """
        result = [0 for _ in range(len(list1) + len(list2) - 1)]
        for i in range(len(list1)):
            for j in range(len(list2)):
                result[i + j] += list1[i] * list2[j]
        return result

    def list_add(self, list1, list2, sub=1):
        """
        Return the element wise addition of two lists
        :param list1: A list of `float` values.
        :param list2: A list of `float` values.
        :param sub: An `int` value to multiply each element in the second list by.
                    Default is set to 1, and setting to -1 results in subtraction.
        :return: A list of `float` values containing the result of the addition.
        """
        return [i + (sub * j) for i, j in zip(list1, list2)]

    def determinant_equation(self, matrix, excluded=[1, 0]):
        """
        Return the equation describing the determinant in terms of some unknown
        variable. The index of each element in the list represents the power of the
        unknown variable. For example, [1, 2, 3] corresponds to the equation
        1 + 2x + 3x^2.
        :param matrix: A list of lists of `float` values.
        :param excluded: A list of `float` values which refers to the value at the
                         row and column along which elements were crossed out.
        :return: A list of `float` values corresponding to an equation (as
                 described above)
        """
        dimensions = self.get_dimensions(matrix)
        if dimensions == [2, 2]:
            tmp = self.list_add(self.list_multiply(matrix[0][0], matrix[1][1]), self.list_multiply(matrix[0][1], matrix[1][0]), sub=-1)
            return self.list_multiply(tmp, excluded)
        else:
            new_matrices = []
            excluded = []
            exclude_row = 0
            for exclude_column in range(dimensions[1]):
                tmp = []
                excluded.append(matrix[exclude_row][exclude_column])
                for row in range(1, dimensions[0]):
                    tmp_row = []
                    for column in range(dimensions[1]):
                        if (row != exclude_row) and (column != exclude_column):
                            tmp_row.append(matrix[row][column])
                    tmp.append(tmp_row)
                new_matrices.append(tmp)
            determinant_equations = [self.determinant_equation(new_matrices[j],
                                                          excluded[j]) for j in range(len(new_matrices))]
            dt_equation = [sum(i) for i in zip(*determinant_equations)]
            return dt_equation

    def identity_matrix(self, dimensions):
        """
        Return an identity matrix of any given dimensions.
        :param dimensions: A list of `float` values.
        :return: A list of lists of `float` values containing the identity matrix.
        """
        matrix = [[0 for j in range(dimensions[1])] for i in range(dimensions[0])]
        for i in range(dimensions[0]):
            matrix[i][i] = 1
        return matrix

    def characteristic_equation(self, matrix):
        """
        Return the characteristic equation of a matrix.
        :param matrix: A list of lists of `float` values.
        :return: A list of lists of `float` values containing the characteristic
                 equation.
        """
        dimensions = self.get_dimensions(matrix)
        return [[[a, -b] for a, b in zip(i, j)] for i, j in zip(matrix, self.identity_matrix(dimensions))]

    def find_eigenvalues(self, matrix):
        """
        Return the eigenvalues of a matrix.
        :param matrix: A list of lists of `float` values.
        :return: A numpy array of `float` values containing the eigenvalues.
        """
        dt_equation = self.determinant_equation(self.characteristic_equation(matrix))
        return np.roots(dt_equation[::-1])

    def pretty_print_results(self):
        line = "*.+'" * 42
        print("[INFO] Eigenvalues: ", self.find_eigenvalues(self.dataset))
        print(line)

    def __str__(self):
        return f"Calculated the eigenvalues and vectors of {self.dataset}"

if __name__ == "__main__":
    A = [[6, 1, -1],
         [0, 7, 0],
         [3, -1, 2]]
    eigenvalues = EigenValuesVectors(A)
    print(eigenvalues)
