
**filter_im**
=============

The ``filter_im`` module contains functions that pre-process XCT images like segmentation and cropping, allowing for 
pore structure analysis. This is mainly used as a quick method of processing raw XCT scans, but Pore2Chip operates 
best on already segmented data, as the methods of segmentation used here rely on Otsu's filtering, which may not capture 
solid matter of a particular density (ex. plant roots).

----

filter_single()
---------------

.. autofunction:: pore2chip.filter_im.filter_single

..

    The coordinates for cropx and cropy begin at the top left.
    
----

filter_list()
-------------

.. autofunction:: pore2chip.filter_im.filter_list

..

    The coordinates for cropx and cropy begin at the top left.

----

read_and_filter()
-----------------

.. autofunction:: pore2chip.filter_im.read_and_filter

..

    The coordinates for cropx and cropy begin at the top left.

----

read_and_filter_list()
----------------------

.. autofunction:: pore2chip.filter_im.read_and_filter_list

..

    The coordinates for cropx and cropy begin at the top left.
    
    Leave crop_depth empty to include all files in directory.

.. note::

   This project is under active development.
