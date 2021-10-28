install
=======

create conda
------------

首先，创建一个虚拟环境

.. code-block:: python

    conda create -n deepke python=3.8
    conda activate deepke

install by pypi
---------------

可以通过pip直接安装所需要的包

.. code-block:: python

    pip install deepke


install by setup.py
-------------------

可以根据需要修改源码或者setup.py中的配置，在进行安装

.. code-block:: python

    python setup.py install