#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;

#ifndef PYSHT_VERSION
#define PYSHT_VERSION "0+unknown"
#endif

namespace {

using InputMatrix = nb::ndarray<nb::numpy, const double, nb::ndim<2>,
                                nb::c_contig, nb::device::cpu>;
using OutputMatrix = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig,
                                 nb::device::cpu>;

OutputMatrix pairwise_distances_impl(const InputMatrix &x, const InputMatrix &y,
                                     bool squared) {
  const std::size_t n_x = x.shape(0);
  const std::size_t n_y = y.shape(0);
  const std::size_t n_features = x.shape(1);

  if (n_features != y.shape(1)) {
    throw std::invalid_argument(
        "x and y must have the same number of features");
  }
  if (n_features == 0) {
    throw std::invalid_argument("x and y must contain at least one feature");
  }
  if (n_y != 0 && n_x > std::numeric_limits<std::size_t>::max() / n_y) {
    throw std::overflow_error("the requested distance matrix is too large");
  }

  const std::size_t output_size = n_x * n_y;
  if (output_size > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    throw std::overflow_error("the requested distance matrix is too large");
  }

  auto output = std::make_unique<double[]>(output_size);
  const double *x_data = x.data();
  const double *y_data = y.data();

  {
    nb::gil_scoped_release release;

    const std::size_t x_size = n_x * n_features;
    const std::size_t y_size = n_y * n_features;
    for (std::size_t i = 0; i < x_size; ++i) {
      if (!std::isfinite(x_data[i])) {
        throw std::invalid_argument("x must contain only finite values");
      }
    }
    for (std::size_t i = 0; i < y_size; ++i) {
      if (!std::isfinite(y_data[i])) {
        throw std::invalid_argument("y must contain only finite values");
      }
    }

    for (std::size_t i = 0; i < n_x; ++i) {
      const double *x_row = x_data + i * n_features;
      for (std::size_t j = 0; j < n_y; ++j) {
        const double *y_row = y_data + j * n_features;
        double distance = 0.0;
        for (std::size_t k = 0; k < n_features; ++k) {
          const double difference = x_row[k] - y_row[k];
          if (squared) {
            distance += difference * difference;
          } else {
            // Iterated hypot avoids the avoidable intermediate overflow and
            // underflow of sqrt(sum(difference * difference)).
            distance = std::hypot(distance, difference);
          }
        }
        if (!std::isfinite(distance)) {
          throw std::overflow_error(
              "a pairwise distance overflowed double precision");
        }
        output[i * n_y + j] = distance;
      }
    }
  }

  double *output_data = output.release();
  nb::capsule owner(output_data, [](void *data) noexcept {
    delete[] static_cast<double *>(data);
  });
  const std::size_t shape[2] = {n_x, n_y};
  return OutputMatrix(output_data, 2, shape, owner);
}

OutputMatrix pairwise_squared_distances(const InputMatrix &x,
                                        const InputMatrix &y) {
  return pairwise_distances_impl(x, y, true);
}

OutputMatrix pairwise_distances(const InputMatrix &x, const InputMatrix &y) {
  return pairwise_distances_impl(x, y, false);
}

nb::dict build_info() {
  nb::dict info;
  info["version"] = PYSHT_VERSION;
  info["cpp_standard"] = 17;
  info["stable_abi"] = true;
  return info;
}

} // namespace

NB_MODULE(_core, module) {
  module.doc() = "Native kernels for pySHT.";
  module.attr("__version__") = PYSHT_VERSION;

  module.def("build_info", &build_info,
             "Return metadata about the compiled pySHT core.");
  module.def(
      "pairwise_squared_distances", &pairwise_squared_distances,
      nb::arg("x").noconvert(), nb::arg("y").noconvert(),
      "Compute squared Euclidean distances between rows of two C-contiguous "
      "float64 matrices.");
  module.def(
      "pairwise_distances", &pairwise_distances, nb::arg("x").noconvert(),
      nb::arg("y").noconvert(),
      "Compute Euclidean distances between rows of two C-contiguous float64 "
      "matrices using overflow-resistant accumulation.");
}
