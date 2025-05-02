from setuptools import setup, find_packages



setup(
        name='weightedkjnn',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/WeightedKNNClassifier',
        description="Weighted KNN Classifier",
        packages=find_packages(include=[
                'weightedknn',
                'weightedknn.*',
                ]),
        install_requires=[ 
                'numpy>=1.22.4',
                'joblib',
                'scikit-learn>=1.2.2',
        ],
        test_suite='test'
        )
