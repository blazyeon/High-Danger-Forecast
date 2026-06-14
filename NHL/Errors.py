"""
Custom exceptions and error handling utilities for NHL statistics application.
Enhanced with better retry logic, validation, and error recovery.
"""
from typing import Optional, Callable, Any, TypeVar, List
import functools
import logging
from datetime import datetime
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')

# ===================== CUSTOM EXCEPTIONS =====================

class NHLAppError(Exception):
    """Base exception for NHL application"""
    pass


class APIError(NHLAppError):
    """Raised when external API calls fail"""
    def __init__(self, service: str, message: str, original_error: Optional[Exception] = None):
        self.service = service
        self.original_error = original_error
        super().__init__(f"{service} API Error: {message}")


class DataValidationError(NHLAppError):
    """Raised when data validation fails"""
    pass


class CacheError(NHLAppError):
    """Raised when cache operations fail"""
    pass


class SimulationError(NHLAppError):
    """Raised when simulation encounters issues"""
    pass


class DateValidationError(DataValidationError):
    """Raised when date validation fails"""
    pass


class SeasonDataError(NHLAppError):
    """Raised when season data is unavailable or invalid"""
    def __init__(self, season: str, message: str):
        self.season = season
        super().__init__(f"Season {season}: {message}")


# ===================== SAFE API CALL WRAPPER =====================

def safe_api_call(
    func: Callable[..., T],
    *args,
    service_name: str = "API",
    fallback: Optional[T] = None,
    raise_on_error: bool = False,
    log_errors: bool = True,
    **kwargs
) -> Optional[T]:
    """
    Safely execute an API call with error handling and logging.
    
    Args:
        func: Function to call
        service_name: Name of the service for logging
        fallback: Value to return on error
        raise_on_error: Whether to raise exception or return fallback
        log_errors: Whether to log errors
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result of func or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"{service_name} call failed: {str(e)}", exc_info=True)
        if raise_on_error:
            raise APIError(service_name, str(e), e)
        return fallback


# ===================== RETRY DECORATOR =====================

def retry_on_failure(
    max_attempts: int = 3,
    backoff_base: float = 0.75,
    backoff_max: float = 10.0,
    exceptions: tuple = (Exception,),
    jitter: bool = True
):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_base: Base delay for exponential backoff
        backoff_max: Maximum delay between retries
        exceptions: Tuple of exceptions to catch
        jitter: Add random jitter to backoff delay
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(backoff_base * (2 ** attempt), backoff_max)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay += random.uniform(0, delay * 0.1)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed, "
                        f"retrying in {delay:.1f}s... Error: {str(e)}"
                    )
                    time.sleep(delay)
            
            return func(*args, **kwargs)  # Should never reach here
        return wrapper
    return decorator


# ===================== VALIDATION FUNCTIONS =====================

def validate_date_range(
    date_str: str,
    min_lookback_days: int = 365 * 3,
    max_future_days: int = 14
) -> datetime:
    """
    Validate date is within acceptable range.
    
    Args:
        date_str: ISO format date string
        min_lookback_days: Maximum days in the past
        max_future_days: Maximum days in the future
    
    Returns:
        Validated datetime object
    
    Raises:
        DateValidationError: If date is invalid or out of range
    """
    from datetime import datetime, timedelta
    
    try:
        date = datetime.fromisoformat(date_str).date()
    except ValueError as e:
        raise DateValidationError(f"Invalid date format: {date_str}") from e
    
    today = datetime.now().date()
    min_date = today - timedelta(days=min_lookback_days)
    max_date = today + timedelta(days=max_future_days)
    
    if date < min_date:
        raise DateValidationError(
            f"Date {date_str} is too far in the past (min: {min_date})"
        )
    if date > max_date:
        raise DateValidationError(
            f"Date {date_str} is too far in the future (max: {max_date})"
        )
    
    return datetime.fromisoformat(date_str)


def safe_division(
    numerator: float, 
    denominator: float, 
    default: float = 0.0,
    min_denominator: float = 1e-10
) -> float:
    """
    Safely divide two numbers, returning default on division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        min_denominator: Minimum denominator threshold
    
    Returns:
        Division result or default
    """
    if abs(denominator) < min_denominator or not isinstance(denominator, (int, float)):
        return default
    
    try:
        result = numerator / denominator
        
        # Check for NaN or infinity
        if not isinstance(result, (int, float)) or result != result or abs(result) == float('inf'):
            return default
        
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def validate_dataframe(
    df: Any,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 0,
    name: str = "DataFrame",
    allow_empty: bool = False
) -> bool:
    """
    Validate a DataFrame meets requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        name: Name for logging
        allow_empty: Whether to allow empty DataFrames
    
    Returns:
        True if valid
    
    Raises:
        DataValidationError: If validation fails
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"{name} is not a DataFrame (type: {type(df)})")
    
    if df.empty:
        if allow_empty:
            return True
        if min_rows > 0:
            raise DataValidationError(f"{name} is empty (requires min {min_rows} rows)")
    
    if len(df) < min_rows:
        raise DataValidationError(
            f"{name} has {len(df)} rows (requires min {min_rows})"
        )
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"{name} missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )
    
    return True


def validate_season_key(season_key: str) -> bool:
    """
    Validate season key format (YYYYYYYY).
    
    Args:
        season_key: Season key to validate
    
    Returns:
        True if valid
    
    Raises:
        DataValidationError: If format is invalid
    """
    if not isinstance(season_key, str):
        raise DataValidationError(f"Season key must be string, got {type(season_key)}")
    
    if len(season_key) != 8:
        raise DataValidationError(
            f"Season key must be 8 digits (YYYYYYYY), got {len(season_key)}: {season_key}"
        )
    
    if not season_key.isdigit():
        raise DataValidationError(
            f"Season key must contain only digits, got: {season_key}"
        )
    
    try:
        start_year = int(season_key[:4])
        end_year = int(season_key[4:])
        
        if end_year != start_year + 1:
            raise DataValidationError(
                f"Invalid season span: {start_year}-{end_year} (must be consecutive years)"
            )
        
        # Reasonable year range
        if not (2000 <= start_year <= 2050):
            raise DataValidationError(
                f"Season year {start_year} out of reasonable range (2000-2050)"
            )
    except ValueError as e:
        raise DataValidationError(f"Could not parse season years: {season_key}") from e
    
    return True


# ===================== PERFORMANCE LOGGING =====================

def log_performance(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        import time
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            
            # Log with different levels based on execution time
            if elapsed > 5.0:
                logger.warning(f"{func.__name__} took {elapsed:.2f}s (slow)")
            elif elapsed > 1.0:
                logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            else:
                logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
            
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
            raise
    return wrapper


# ===================== CONTEXT MANAGERS =====================

class ErrorContext:
    """Context manager for handling errors with custom messages"""
    
    def __init__(self, operation: str, raise_on_error: bool = False):
        self.operation = operation
        self.raise_on_error = raise_on_error
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error during {self.operation}: {exc_val}", exc_info=True)
            
            if self.raise_on_error:
                return False  # Re-raise exception
            else:
                return True  # Suppress exception


# ===================== EXPORTS =====================

__all__ = [
    'NHLAppError',
    'APIError',
    'DataValidationError',
    'CacheError',
    'SimulationError',
    'DateValidationError',
    'SeasonDataError',
    'safe_api_call',
    'retry_on_failure',
    'validate_date_range',
    'safe_division',
    'validate_dataframe',
    'validate_season_key',
    'log_performance',
    'ErrorContext',
]