"""
Validation utilities for input validation and data integrity checks.
"""

import re
import string
from typing import Any, Dict, List, Optional, Union, Callable, Type
from pathlib import Path
import json
import yaml

from ..core.exceptions import ValidationError

class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool = True, errors: Optional[List[str]] = None, 
                 warnings: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str) -> 'ValidationResult':
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
        return self
    
    def add_warning(self, warning: str) -> 'ValidationResult':
        """Add a warning to the result."""
        self.warnings.append(warning)
        return self
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
        return self
    
    def raise_if_invalid(self, context: Optional[str] = None) -> None:
        """Raise ValidationError if result is invalid."""
        if not self.is_valid:
            message = f"Validation failed: {'; '.join(self.errors)}"
            if context:
                message = f"{context}: {message}"
            raise ValidationError(message, details={
                'errors': self.errors,
                'warnings': self.warnings
            })
    
    def __bool__(self) -> bool:
        """Boolean representation."""
        return self.is_valid
    
    def __str__(self) -> str:
        """String representation."""
        if self.is_valid:
            status = "Valid"
            if self.warnings:
                status += f" (with {len(self.warnings)} warnings)"
            return status
        else:
            return f"Invalid: {len(self.errors)} errors, {len(self.warnings)} warnings"

class Validator:
    """Base validator class."""
    
    def __init__(self, name: str = "Validator"):
        self.name = name
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate a value. Override in subclasses."""
        return ValidationResult()
    
    def __call__(self, value: Any) -> ValidationResult:
        """Make validator callable."""
        return self.validate(value)

class StringValidator(Validator):
    """Validator for string values."""
    
    def __init__(self, 
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 pattern: Optional[str] = None,
                 allowed_chars: Optional[str] = None,
                 forbidden_chars: Optional[str] = None,
                 strip_whitespace: bool = True,
                 name: str = "StringValidator"):
        super().__init__(name)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
        self.allowed_chars = set(allowed_chars) if allowed_chars else None
        self.forbidden_chars = set(forbidden_chars) if forbidden_chars else None
        self.strip_whitespace = strip_whitespace
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate string value."""
        result = ValidationResult()
        
        # Type check
        if not isinstance(value, str):
            return result.add_error(f"Expected string, got {type(value).__name__}")
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            value = value.strip()
        
        # Length checks
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(f"String too short: {len(value)} < {self.min_length}")
        
        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(f"String too long: {len(value)} > {self.max_length}")
        
        # Pattern check
        if self.pattern and not self.pattern.match(value):
            result.add_error(f"String does not match pattern: {self.pattern.pattern}")
        
        # Character restrictions
        if self.allowed_chars:
            invalid_chars = set(value) - self.allowed_chars
            if invalid_chars:
                result.add_error(f"Contains forbidden characters: {invalid_chars}")
        
        if self.forbidden_chars:
            found_forbidden = set(value) & self.forbidden_chars
            if found_forbidden:
                result.add_error(f"Contains forbidden characters: {found_forbidden}")
        
        return result

class NumericValidator(Validator):
    """Validator for numeric values."""
    
    def __init__(self,
                 min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 allow_negative: bool = True,
                 allow_zero: bool = True,
                 integer_only: bool = False,
                 name: str = "NumericValidator"):
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_negative = allow_negative
        self.allow_zero = allow_zero
        self.integer_only = integer_only
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate numeric value."""
        result = ValidationResult()
        
        # Type check
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return result.add_error(f"Cannot convert to number: {value}")
        
        # Integer check
        if self.integer_only and not isinstance(value, int) and not value.is_integer():
            result.add_error(f"Expected integer, got {value}")
        
        # Range checks
        if self.min_value is not None and value < self.min_value:
            result.add_error(f"Value too small: {value} < {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            result.add_error(f"Value too large: {value} > {self.max_value}")
        
        # Sign checks
        if not self.allow_negative and value < 0:
            result.add_error(f"Negative values not allowed: {value}")
        
        if not self.allow_zero and value == 0:
            result.add_error("Zero not allowed")
        
        return result

class ListValidator(Validator):
    """Validator for list values."""
    
    def __init__(self,
                 item_validator: Optional[Validator] = None,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 unique_items: bool = False,
                 name: str = "ListValidator"):
        super().__init__(name)
        self.item_validator = item_validator
        self.min_length = min_length
        self.max_length = max_length
        self.unique_items = unique_items
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate list value."""
        result = ValidationResult()
        
        # Type check
        if not isinstance(value, list):
            return result.add_error(f"Expected list, got {type(value).__name__}")
        
        # Length checks
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(f"List too short: {len(value)} < {self.min_length}")
        
        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(f"List too long: {len(value)} > {self.max_length}")
        
        # Uniqueness check
        if self.unique_items and len(set(value)) != len(value):
            result.add_error("List contains duplicate items")
        
        # Validate individual items
        if self.item_validator:
            for i, item in enumerate(value):
                item_result = self.item_validator.validate(item)
                if not item_result.is_valid:
                    for error in item_result.errors:
                        result.add_error(f"Item {i}: {error}")
        
        return result

class DictValidator(Validator):
    """Validator for dictionary values."""
    
    def __init__(self,
                 required_keys: Optional[List[str]] = None,
                 optional_keys: Optional[List[str]] = None,
                 key_validators: Optional[Dict[str, Validator]] = None,
                 allow_extra_keys: bool = True,
                 name: str = "DictValidator"):
        super().__init__(name)
        self.required_keys = set(required_keys or [])
        self.optional_keys = set(optional_keys or [])
        self.key_validators = key_validators or {}
        self.allow_extra_keys = allow_extra_keys
    
    def validate(self, value: Any) -> ValidationResult:
        """Validate dictionary value."""
        result = ValidationResult()
        
        # Type check
        if not isinstance(value, dict):
            return result.add_error(f"Expected dict, got {type(value).__name__}")
        
        keys = set(value.keys())
        
        # Check required keys
        missing_keys = self.required_keys - keys
        if missing_keys:
            result.add_error(f"Missing required keys: {missing_keys}")
        
        # Check for extra keys
        if not self.allow_extra_keys:
            allowed_keys = self.required_keys | self.optional_keys
            extra_keys = keys - allowed_keys
            if extra_keys:
                result.add_error(f"Extra keys not allowed: {extra_keys}")
        
        # Validate individual values
        for key, key_value in value.items():
            if key in self.key_validators:
                key_result = self.key_validators[key].validate(key_value)
                if not key_result.is_valid:
                    for error in key_result.errors:
                        result.add_error(f"Key '{key}': {error}")
        
        return result

# Prompt validation functions
def validate_prompt(prompt: str, 
                   min_length: int = 1,
                   max_length: int = 8192,
                   allow_empty: bool = False) -> ValidationResult:
    """
    Validate a prompt string.
    
    Args:
        prompt: Prompt to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        allow_empty: Whether to allow empty prompts
        
    Returns:
        ValidationResult
    """
    
    validator = StringValidator(
        min_length=0 if allow_empty else min_length,
        max_length=max_length,
        strip_whitespace=True,
        name="PromptValidator"
    )
    
    result = validator.validate(prompt)
    
    # Additional prompt-specific checks
    if result.is_valid and prompt.strip():
        # Check for potentially harmful content
        if any(pattern in prompt.lower() for pattern in ['<script>', 'javascript:', 'eval(']):
            result.add_warning("Prompt contains potentially harmful content")
        
        # Check for very repetitive content
        words = prompt.split()
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:
                result.add_warning("Prompt appears to be very repetitive")
    
    return result

def validate_config(config: Dict[str, Any], 
                   schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration to validate
        schema: Validation schema
        
    Returns:
        ValidationResult
    """
    
    result = ValidationResult()
    
    # Check required fields
    required = schema.get('required', [])
    for field in required:
        if field not in config:
            result.add_error(f"Required field missing: {field}")
    
    # Validate each field
    properties = schema.get('properties', {})
    for field, field_config in properties.items():
        if field in config:
            field_result = _validate_config_field(config[field], field_config, field)
            result.merge(field_result)
    
    # Check for extra fields if not allowed
    if not schema.get('additionalProperties', True):
        allowed_fields = set(properties.keys())
        extra_fields = set(config.keys()) - allowed_fields
        if extra_fields:
            result.add_error(f"Extra fields not allowed: {extra_fields}")
    
    return result

def _validate_config_field(value: Any, 
                          field_config: Dict[str, Any], 
                          field_name: str) -> ValidationResult:
    """Validate a single configuration field."""
    
    result = ValidationResult()
    
    # Type validation
    expected_type = field_config.get('type')
    if expected_type:
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            result.add_error(f"Field '{field_name}': expected {expected_type}, got {type(value).__name__}")
            return result
    
    # String validation
    if isinstance(value, str):
        min_length = field_config.get('minLength')
        max_length = field_config.get('maxLength')
        pattern = field_config.get('pattern')
        
        validator = StringValidator(
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            name=f"Field_{field_name}"
        )
        result.merge(validator.validate(value))
    
    # Numeric validation
    elif isinstance(value, (int, float)):
        minimum = field_config.get('minimum')
        maximum = field_config.get('maximum')
        
        validator = NumericValidator(
            min_value=minimum,
            max_value=maximum,
            name=f"Field_{field_name}"
        )
        result.merge(validator.validate(value))
    
    # Array validation
    elif isinstance(value, list):
        min_items = field_config.get('minItems')
        max_items = field_config.get('maxItems')
        unique_items = field_config.get('uniqueItems', False)
        
        validator = ListValidator(
            min_length=min_items,
            max_length=max_items,
            unique_items=unique_items,
            name=f"Field_{field_name}"
        )
        result.merge(validator.validate(value))
    
    # Enum validation
    enum_values = field_config.get('enum')
    if enum_values and value not in enum_values:
        result.add_error(f"Field '{field_name}': value must be one of {enum_values}")
    
    return result

class ValidationMixin:
    """Mixin class that provides validation capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_schema: Optional[Dict[str, Any]] = None
        self._validators: Dict[str, Validator] = {}
    
    def set_validation_schema(self, schema: Dict[str, Any]) -> None:
        """Set validation schema for this object."""
        self._validation_schema = schema
    
    def add_validator(self, field: str, validator: Validator) -> None:
        """Add a validator for a specific field."""
        self._validators[field] = validator
    
    def validate_field(self, field: str, value: Any) -> ValidationResult:
        """Validate a single field."""
        if field in self._validators:
            return self._validators[field].validate(value)
        
        if self._validation_schema and 'properties' in self._validation_schema:
            field_config = self._validation_schema['properties'].get(field)
            if field_config:
                return _validate_config_field(value, field_config, field)
        
        return ValidationResult()  # No validation rules, assume valid
    
    def validate_all_fields(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate all fields in a data dictionary."""
        result = ValidationResult()
        
        for field, value in data.items():
            field_result = self.validate_field(field, value)
            if not field_result.is_valid:
                result.merge(field_result)
        
        return result
    
    def validate_self(self) -> ValidationResult:
        """Validate this object's current state."""
        if hasattr(self, '__dict__'):
            return self.validate_all_fields(self.__dict__)
        return ValidationResult()

# File validation utilities
def validate_file_path(path: Union[str, Path],
                      must_exist: bool = True,
                      must_be_file: bool = True,
                      allowed_extensions: Optional[List[str]] = None) -> ValidationResult:
    """Validate a file path."""
    
    result = ValidationResult()
    path = Path(path)
    
    if must_exist and not path.exists():
        result.add_error(f"Path does not exist: {path}")
        return result
    
    if must_be_file and path.exists() and not path.is_file():
        result.add_error(f"Path is not a file: {path}")
    
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            result.add_error(f"File extension not allowed: {path.suffix} (allowed: {allowed_extensions})")
    
    return result

def validate_json_file(path: Union[str, Path]) -> ValidationResult:
    """Validate that a file contains valid JSON."""
    
    result = validate_file_path(path, allowed_extensions=['.json'])
    
    if result.is_valid:
        try:
            with open(path, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            result.add_error(f"Invalid JSON: {e}")
        except Exception as e:
            result.add_error(f"Error reading file: {e}")
    
    return result

def validate_yaml_file(path: Union[str, Path]) -> ValidationResult:
    """Validate that a file contains valid YAML."""
    
    result = validate_file_path(path, allowed_extensions=['.yaml', '.yml'])
    
    if result.is_valid:
        try:
            with open(path, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            result.add_error(f"Invalid YAML: {e}")
        except Exception as e:
            result.add_error(f"Error reading file: {e}")
    
    return result