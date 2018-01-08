import qtrader


def valid_type(variable, correct_type):
    """Validate type of `variable`.

    Parameters
    ----------
    variable: object
        Variable to be type-validated
    correct_type: type
        Correct/Expected type of `variable`

    Raises
    ------
    error: TypeError
        Mismatched type

    Examples
    --------
    >>> qtrader.framework.VALID_TYPE = True
    >>> a = 5.0
    >>> qtrader.utils.valid_type(a, int)
    >>> TypeError: invalid `5.0` type; passed type: <class 'float'>; expected type: <class 'int'>"""
    if qtrader.framework.VALID_TYPE:
        if not isinstance(variable, correct_type):
            raise TypeError(
                'invalid `%s` type; passed type: %s; expected type: %s' % (
                    variable, type(variable), correct_type)
            )
    qtrader.framework.logger.debug(
        'successful valid_type(variable, correct_type) call')


def valid_shape(variable, correct_shape):
    """Validate shape of `variable`.

    Parameters
    ----------
    variable: object
        Variable to be shape-validated
    correct_shape: object | tuple
        Correct/Expected shape of `variable` or struct with correct shape

    Raises
    ------
    error: AttributeError | ValueError
        Mismatched shape

    Examples
    --------
    >>> qtrader.framework.VALID_SHAPE = True
    >>> a = np.arange(3).reshape(-1, 1)
    >>> qtrader.utils.valid_shape(a, (3,))
    >>> 'invalid `[[0, 1, 2]]` shape;  passed shape: (3, 1); expected shape: (3,)'
    """
    if qtrader.framework.VALID_SHAPE:
        if not hasattr(variable, 'shape'):
            raise AttributeError(
                '`%s` has no attribute `shape`.' % (variable)
            )
        if hasattr(correct_shape, 'shape'):
            if correct_shape.shape != variable.shape:
                raise ValueError(
                    'invalid `%s` shape;  passed shape: %s; expected shape: %s' % (
                        variable, variable.shape, correct_shape.shape)
                )
        else:
            if variable.shape != correct_shape:
                raise ValueError(
                    'invalid `%s` shape;  passed shape: %s; expected shape: %s' % (
                        variable, variable.shape, correct_shape)
                )
    qtrader.framework.logger.debug(
        'successful valid_shape(variable, correct_shape) call')
