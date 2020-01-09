# -*- coding: utf-8 -*-
"""
Created on 2020-01-09 20:50:29

@author: Carsten Knoll
"""

import unittest
import torch
import numpy as np

from .. import matrix_approx as ma


class TestHHMatrix(unittest.TestCase):
    def setUp(self):
        pass

    def test_hh(self):
        n = 5
        u = torch.rand(3)

        hh = ma.hh_matix(n, u)
        self.assertTrue(torch.all(hh == hh.T))
        self.assertTrue(torch.allclose(hh@hh, torch.eye(n), atol=1e-6))


class TestSVDMatrix(unittest.TestCase):
    def setUp(self):
        pass

    def test_prediction(self):
        n = 5
        N_data = 10

        X_train = torch.rand(n, N_data)
        A = torch.rand(n, n)
        Y_train = A@X_train

        net = ma.Net(n)
        # test for unwanted nan's
        self.assertFalse(np.isnan(net.matrix.matrix.data.numpy()).any())
