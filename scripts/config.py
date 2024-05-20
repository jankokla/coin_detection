"""config.py: Helper constants to import."""

ID_TO_CCY = {
    0: 'CHF',
    1: 'EUR',
    2: 'OOD'
}

ID_TO_SIDE = {
    0: 'tail',
    1: 'head'
}

ID_TO_CHF_IMG = {
    0: '5CHF',
    1: '2CHF/1CHF/0.5CHF',
    2: '0.2CHF/0.1CHF/0.05CHF'
}

ID_TO_EUR = {
    0: '2EUR',
    1: '1EUR',
    2: '0.5EUR',
    3: '0.2EUR',
    4: '0.1EUR',
    5: '0.05EUR',
    6: '0.02EUR',
    7: '0.01EUR'
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

example_row = [
    '5CHF', '2CHF', '1CHF', '0.5CHF', '0.2CHF', '0.1CHF', '0.05CHF', '2EUR',
    '1EUR', '0.5EUR', '0.2EUR', '0.1EUR', '0.05EUR', '0.02EUR', '0.01EUR', 'OOD'
]

size_dict = {
    '5CHF': 31.45,
    '2CHF': 27.40,
    '1CHF': 23.20,
    '0.5CHF': 18.20,
    '0.2CHF': 21.05,
    '0.1CHF': 19.15,
    '0.05CHF': 17.15,
    '2EUR': 25.75,
    '1EUR': 23.25,
    '0.5EUR': 24.25,
    '0.2EUR': 22.25,
    '0.1EUR': 19.75,
    '0.05EUR': 21.25,
    '0.02EUR': 18.75,
    '0.01EUR': 16.25,
}
