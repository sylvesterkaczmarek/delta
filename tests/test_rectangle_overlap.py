# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that valid output tiles partition odd overlaps without duplicate pixels."""

import copy
import unittest

import numpy as np

from delta.imagery.rectangle import Rectangle


def absolute_tiles(tiles, by_block):
    if not by_block:
        return tiles
    result = []
    for block, children in tiles:
        for child in children:
            tile = copy.copy(child)
            tile.shift(block.min_x, block.min_y)
            result.append(tile)
    return result


class TestRectangleOverlap(unittest.TestCase):
    def assert_partition(self, rectangle, tile_shape, overlap, **options):
        original = rectangle.bounds()
        tiles, valid = rectangle.make_tile_rois(
            tile_shape, overlap_shape=overlap, **options
        )
        by_block = options.get("by_block", False)
        tiles = absolute_tiles(tiles, by_block)
        valid = absolute_tiles(valid, by_block)
        counts = np.zeros((rectangle.height(), rectangle.width()), dtype=int)
        self.assertEqual(len(tiles), len(valid))
        for tile, core in zip(tiles, valid):
            self.assertTrue(tile.contains_rect(core))
            self.assertTrue(rectangle.contains_rect(core))
            self.assertTrue(core.has_area())
            counts[
                core.min_y - rectangle.min_y : core.max_y - rectangle.min_y,
                core.min_x - rectangle.min_x : core.max_x - rectangle.min_x,
            ] += 1
        np.testing.assert_array_equal(counts, np.ones_like(counts))
        self.assertEqual(rectangle.bounds(), original)
        return tiles, valid

    def test_odd_overlap_partitions_partial_and_shifted_tiles(self):
        overlaps = ((1, 0), (0, 1), (3, 1), (1, 3), (3, 3), (2, 1), (3, 2))
        for overlap in overlaps:
            for by_block in (False, True):
                for include_partials in (False, True):
                    with self.subTest(
                        overlap=overlap,
                        by_block=by_block,
                        include_partials=include_partials,
                    ):
                        self.assert_partition(
                            Rectangle(-5, 10, 6, 19),
                            (6, 5),
                            overlap,
                            by_block=by_block,
                            include_partials=include_partials,
                            partials_overlap=not include_partials,
                        )

    def test_adjacent_unique_boundaries_meet_at_the_same_pixel(self):
        tiles, cores = Rectangle(0, 0, 11, 5).make_tile_rois((6, 5), (3, 0))
        self.assertEqual(
            [tile.bounds() for tile in tiles],
            [(0, 6, 0, 5), (3, 9, 0, 5), (6, 11, 0, 5)],
        )
        self.assertEqual(
            [core.bounds() for core in cores],
            [(0, 4, 0, 5), (4, 7, 0, 5), (7, 11, 0, 5)],
        )

    def test_output_is_independent_of_tile_processing_order(self):
        rectangle = Rectangle(0, 0, 11, 9)
        tiles, cores = rectangle.make_tile_rois((6, 5), (3, 1))
        forward = np.zeros((9, 11), dtype=int)
        reverse = forward.copy()
        indexed = list(enumerate(zip(tiles, cores), start=1))
        for output, iteration in ((forward, indexed), (reverse, reversed(indexed))):
            for number, (tile, core) in iteration:
                prediction = np.full((tile.height(), tile.width()), number)
                cropped = prediction[
                    core.min_y - tile.min_y : core.max_y - tile.min_y,
                    core.min_x - tile.min_x : core.max_x - tile.min_x,
                ]
                output[core.min_y : core.max_y, core.min_x : core.max_x] = cropped
        np.testing.assert_array_equal(forward, reverse)
        self.assertTrue((forward > 0).all())

    def test_yx_wrapper_matches_xy_output_including_block_coordinates(self):
        for by_block in (False, True):
            with self.subTest(by_block=by_block):
                rectangle = Rectangle(5, -3, 16, 6)
                xy = rectangle.make_tile_rois((6, 5), (3, 1), by_block=by_block)
                yx = rectangle.make_tile_rois_yx((5, 6), (1, 3), by_block=by_block)
                for direct, transposed in zip(xy, yx):
                    self.assertEqual(
                        [r.bounds() for r in absolute_tiles(direct, by_block)],
                        [r.bounds() for r in absolute_tiles(transposed, by_block)],
                    )
                self.assert_partition(rectangle, (6, 5), (3, 1), by_block=by_block)

    def test_even_and_zero_overlaps_keep_existing_boundaries(self):
        rectangle = Rectangle(0, 0, 10, 10)
        for shape, overlap in (((6, 6), (2, 2)), ((5, 5), (0, 0))):
            with self.subTest(shape=shape, overlap=overlap):
                _, cores = self.assert_partition(rectangle, shape, overlap)
                self.assertEqual(
                    [r.bounds() for r in cores],
                    [(0, 5, 0, 5), (5, 10, 0, 5), (0, 5, 5, 10), (5, 10, 5, 10)],
                )

    def test_single_tile_is_not_trimmed(self):
        for by_block in (False, True):
            with self.subTest(by_block=by_block):
                rectangle = Rectangle(2, 3, 8, 8)
                _, cores = self.assert_partition(
                    rectangle, (6, 5), (3, 1), by_block=by_block
                )
                self.assertEqual([r.bounds() for r in cores], [rectangle.bounds()])

    def test_containing_rectangle_keeps_valid_output_inside_requested_region(self):
        rectangle = Rectangle(2, 2, 13, 11)
        self.assert_partition(
            rectangle,
            (6, 5),
            (3, 1),
            include_partials=False,
            partials_overlap=True,
            containing_rect=Rectangle(0, 0, 20, 20),
        )


if __name__ == "__main__":
    unittest.main()
