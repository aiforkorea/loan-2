# apps/loan/dbmodels.py

from apps.extensions import db
from sqlalchemy import BigInteger, DECIMAL, PrimaryKeyConstraint, ForeignKeyConstraint, ForeignKey
from sqlalchemy.orm import relationship

# apps/dbmodels.py에 정의된 User 모델을 참조
from apps.dbmodels import User 

# --- 1. CustInfo 모델 (고객 기본 정보) ---
class CustInfo(db.Model):
    __tablename__ = 'cust_info'
    # 💡 수정: db.String(8) -> db.Date (날짜 조작 효율성 향상)
    base_dt = db.Column(db.Date, primary_key=True) 
    
    # 💡 수정: String(10) -> Integer, users.username -> users.id 참조
    cust_id = db.Column(db.Integer, ForeignKey('users.id'), primary_key=True)
    
    gender = db.Column(db.String(10), nullable=True)
    married = db.Column(db.String(5), nullable=True)
    education = db.Column(db.String(20), nullable=True)
    self_employed = db.Column(db.String(5), nullable=True)
    
    # 관계: User.id를 참조하므로 primaryjoin 제거하고 단순하게 설정
    user = relationship('User')
    # 내부 관계
    family_members = relationship('FamilyInfo', back_populates='parent_cust', lazy='dynamic')

    def __repr__(self):
        return f"<CustInfo {self.base_dt}/{self.cust_id}>"

# --- 2. FamilyInfo 모델 (가족 정보) ---
class FamilyInfo(db.Model):
    __tablename__ = 'family_info'
    
    # 💡 수정: db.String(8) -> db.Date
    base_dt = db.Column(db.Date, primary_key=True)
    # 💡 수정: cust_id를 Integer로 변경 (CustInfo 참조 타입에 맞춤)
    cust_id = db.Column(db.Integer, primary_key=True)
    family_cust_id = db.Column(db.String(10), primary_key=True)
    
    living_together = db.Column(db.String(5), nullable=True)
    
    # Foreign Key Constraint (CustInfo와의 복합키 참조)
    __table_args__ = (
        ForeignKeyConstraint(
            ['base_dt', 'cust_id'], 
            ['cust_info.base_dt', 'cust_info.cust_id'] # 타입은 자동으로 매칭됨
        ),
        PrimaryKeyConstraint('base_dt', 'cust_id', 'family_cust_id')
    )
    parent_cust = relationship('CustInfo', back_populates='family_members')

    def __repr__(self):
        return f"<FamilyInfo {self.base_dt}/{self.cust_id}/{self.family_cust_id}>"

# --- 3. LoanDefaultAccount 모델 (연체 계좌 정보) ---
class LoanDefaultAccount(db.Model):
    __tablename__ = 'loan_default_account'
    
    loan_account_id = db.Column(db.String(12), primary_key=True)
    
    # 💡 수정: db.String(8) -> db.Date
    registration_date = db.Column(db.Date, nullable=True) 
    registration_time = db.Column(db.String(6), nullable=True)
    loan_default = db.Column(db.String(5), nullable=True)
    
    loan_applicants = relationship('LoanApplicantInfo', back_populates='loan_account', lazy='dynamic')

    def __repr__(self):
        return f"<LoanDefaultAccount {self.loan_account_id}>"

# --- 4. LoanApplicantInfo 모델 (대출 신청 정보) ---
class LoanApplicantInfo(db.Model):
    __tablename__ = 'loan_applicant_info'
    
    applicant_id = db.Column(db.String(10), primary_key=True)
    
    # 💡 수정: db.String(8) -> db.Date
    applicant_date = db.Column(db.Date, nullable=False) 
    applicant_time = db.Column(db.String(6), nullable=False)
    
    # 💡 수정: String(10) -> Integer, users.username -> users.id 참조
    cust_id = db.Column(db.Integer, ForeignKey('users.id'), nullable=False)
    
    applicant_income = db.Column(BigInteger, nullable=True)
    coapplicant_income = db.Column(BigInteger, nullable=True)
    credit_history = db.Column(DECIMAL(5, 2), nullable=True)
    property_area = db.Column(db.String(10), nullable=True)
    loan_amount = db.Column(BigInteger, nullable=True)
    loan_amount_term = db.Column(db.Integer, nullable=True)
    
    # FK: LoanDefaultAccount 참조
    loan_account_id = db.Column(db.String(12), ForeignKey('loan_default_account.loan_account_id'), nullable=True)
    
    # 관계: User.id를 참조하므로 primaryjoin 제거하고 단순하게 설정
    applicant_user = relationship('User')
    # 내부 관계
    loan_account = relationship('LoanDefaultAccount', back_populates='loan_applicants')

    def __repr__(self):
        return f"<LoanApplicantInfo {self.applicant_id}>"