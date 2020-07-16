# Import NumPy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

def test1():
    # Define vector v
    v = np.array([1,1])
    # Plots vector v as blue arrow with red dot at origin (0,0) using Matplotlib

    # Creates axes of plot referenced 'ax'
    ax = plt.axes()

    # Plots red dot at origin (0,0)
    ax.plot(0,0,'or')

    # Plots vector v as blue arrow starting at origin 0,0
    ax.arrow(0, 0, *v, color='b', linewidth=2.0, head_width=0.20, head_length=0.25)

    # Sets limit for plot for x-axis
    plt.xlim(-2,2)

    # Set major ticks for x-axis
    major_xticks = np.arange(-2, 3)
    ax.set_xticks(major_xticks)


    # Sets limit for plot for y-axis
    plt.ylim(-1, 2)

    # Set major ticks for y-axis
    major_yticks = np.arange(-1, 3)
    ax.set_yticks(major_yticks)

    # Creates gridlines for only major tick marks
    plt.grid(b=True, which='major')

    # Displays final plot
    plt.show()


def test2():
    # Makes Python package NumPy available using import method
    import numpy as np

    # Creates matrix t (right side of the augmented matrix).
    t = np.array([4, 11])

    # Creates matrix vw (left side of the augmented matrix).
    vw = np.array([[1, 2], [3, 5]])

    # Prints vw and t
    print("\nMatrix vw:", vw, "\nVector t:", t, sep="\n")


def check_vector_span(set_of_vectors, vector_to_check):
    # Creates an empty vector of correct size
    vector_of_scalars = np.asarray([None]*set_of_vectors.shape[0])

    # Solves for the scalars that make the equation true if vector is within the span
    try:
        # TODO: Use np.linalg.solve() function here to solve for vector_of_scalars
        vector_of_scalars = np.linalg.solve(set_of_vectors, vector_to_check)
        if not (vector_of_scalars is None):
            print("\nVector is within span.\nScalars in s:", vector_of_scalars)
    # Handles the cases when the vector is NOT within the span
    except Exception as exception_type:
        if str(exception_type) == "Singular matrix":
            print("\nNo single solution\nVector is NOT within span")
        else:
            print("\nUnexpected Exception Error:", exception_type)
    return vector_of_scalars


if __name__ == '__main__':
    np.array()
    test2()
