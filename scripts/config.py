"""config.py: Helper constants to import."""

import numpy as np

LABEL_TO_ID = {
    '5CHF': 0,
    '2CHF': 1,
    '1CHF': 2,
    '0.5CHF': 3,
    '0.2CHF': 4,
    '0.1CHF': 5,
    '0.05CHF': 6,
    '2EUR': 7,
    '1EUR': 8,
    '0.5EUR': 9,
    '0.2EUR': 10,
    '0.1EUR': 11,
    '0.05EUR': 12,
    '0.02EUR': 13,
    '0.01EUR': 14,
    'OOD': 15
}

ID_TO_LABEL = {
    0: '5CHF',
    1: '2CHF',
    2: '1CHF',
    3: '0.5CHF',
    4: '0.2CHF',
    5: '0.1CHF',
    6: '0.05CHF',
    7: '2EUR',
    8: '1EUR',
    9: '0.5EUR',
    10: '0.2EUR',
    11: '0.1EUR',
    12: '0.05EUR',
    13: '0.02EUR',
    14: '0.01EUR',
    15: 'OOD'
}

row_template = {
    '5CHF': 0,
    '2CHF': 0,
    '1CHF': 0,
    '0.5CHF': 0,
    '0.2CHF': 0,
    '0.1CHF': 0,
    '0.05CHF': 0,
    '2EUR': 0,
    '1EUR': 0,
    '0.5EUR': 0,
    '0.2EUR': 0,
    '0.1EUR': 0,
    '0.05EUR': 0,
    '0.02EUR': 0,
    '0.01EUR': 0,
    'OOD': 0
}

id_to_label = np.vectorize(lambda x: ID_TO_LABEL.get(x, "Unknown"))

example_row = [
    '5CHF', '2CHF', '1CHF', '0.5CHF', '0.2CHF', '0.1CHF', '0.05CHF', '2EUR',
    '1EUR', '0.5EUR', '0.2EUR', '0.1EUR', '0.05EUR', '0.02EUR', '0.01EUR', 'OOD'
]
