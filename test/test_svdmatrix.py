# -*- coding: utf-8 -*-
"""
Created on 2020-01-09 20:50:29

@author: Carsten Knoll
"""

import unittest
import torch

from .. import matrix_approx as ma


class TestHHMatrix(unittest.TestCase):
    def setUp(self):
        pass

    def test_hh(self):
        n = 5
        u = torch.rand(3)

        hh = ma.hh_matix(n, u)
        self.assertEqual(hh, hh.T)
        self.assertEqual(hh@hh, torch.eye(n))
