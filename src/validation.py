from datetime import datetime, timedelta
from typing import Dict, Optional


class AmountValidator:
    """Validate consistency between numerical and written amounts"""
    
    def validate_amount_consistency(
        self,
        amount_digits: Optional[float],
        amount_words: Optional[float],
        tolerance: float = 0.01
    ) -> Dict:
        """
        Check if numerical amount matches written amount
        
        Args:
            amount_digits: Numerical amount from amount box
            amount_words: Amount parsed from written words
            tolerance: Acceptable difference (default $0.01)
        
        Returns:
            {
                'consistent': bool,
                'difference': float or None,
                'fraud_risk': str,  # 'low', 'medium', 'high', 'critical'
                'message': str,
                'digits': float or None,
                'words': float or None
            }
        """
        # Handle missing data
        if amount_digits is None and amount_words is None:
            return {
                'consistent': False,
                'difference': None,
                'fraud_risk': 'high',
                'message': 'Both amounts missing - requires manual review',
                'digits': None,
                'words': None
            }
        
        if amount_digits is None:
            return {
                'consistent': False,
                'difference': None,
                'fraud_risk': 'medium',
                'message': 'Numerical amount not readable',
                'digits': None,
                'words': amount_words
            }
        
        if amount_words is None:
            return {
                'consistent': False,
                'difference': None,
                'fraud_risk': 'medium',
                'message': 'Written amount not readable',
                'digits': amount_digits,
                'words': None
            }
        
        # Calculate difference
        diff = abs(amount_digits - amount_words)
        
        # Check consistency
        consistent = diff <= tolerance
        
        # Assess fraud risk based on discrepancy magnitude
        if consistent:
            risk = 'low'
            message = f'Amounts match (${amount_digits:.2f})'
        elif diff <= 10:
            risk = 'medium'
            message = f'Minor discrepancy: ${diff:.2f} difference'
        elif diff <= 100:
            risk = 'high'
            message = f'Major discrepancy: ${diff:.2f} - REVIEW REQUIRED'
        else:
            risk = 'critical'
            message = f'FRAUD ALERT: ${diff:.2f} discrepancy - LIKELY FRAUD'
        
        return {
            'consistent': consistent,
            'difference': diff,
            'fraud_risk': risk,
            'message': message,
            'digits': amount_digits,
            'words': amount_words
        }


def validate_aba_routing(routing_number):
    """
    Validates ABA routing number using checksum.
    3-7-1-3-7-1-3-7-1 weighting.
    """
    if not routing_number or len(routing_number) != 9 or not routing_number.isdigit():
        return False
    
    digits = [int(d) for d in routing_number]
    weights = [3, 7, 1, 3, 7, 1, 3, 7]
    
    checksum = sum(d * w for d, w in zip(digits[:8], weights))
    check_digit = (10 - (checksum % 10)) % 10
    
    return check_digit == digits[8]


def validate_check_date(date_str):
    """
    Checks if date is stale (> 6 months old) or post-dated.
    Assumes date_str is parsed to YYYY-MM-DD.
    """
    # Placeholder logic
    # In a real app, we'd need robust date parsing first.
    return {
        "valid": True,
        "message": "Date validation placeholder"
    }


def validate_amount_consistency(amount_digits, amount_words, tolerance=0.01):
    """
    Checks if numeric amount matches written amount.
    Simple wrapper for AmountValidator class.
    """
    validator = AmountValidator()
    result = validator.validate_amount_consistency(amount_digits, amount_words, tolerance)
    
    return {
        "match": result['consistent'],
        "message": result['message'],
        "fraud_risk": result['fraud_risk'],
        "difference": result['difference']
    }
