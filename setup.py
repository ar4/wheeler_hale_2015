from setuptools import setup

setup(
        name='wheeler_hale_2015',
        version='0.1.0',
        description='An implementation of Wheeler and Hale\'s 2015 method for aligning well logs using dynamic warping',
        url='https://github.com/ar4/wheeler_hale_2015',
        author='Alan Richardson',
        license='MIT',
        packages=['wheeler_hale_2015'],
        package_dir={'wheeler_hale_2015':'wheeler_hale_2015'},
        install_requires=['numpy','scipy','pandas','lasio','fastdtw'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ]
)
