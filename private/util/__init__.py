# import hashlib
# import uuid


# class PasswordManager:
#     @staticmethod
#     def validate_password(hashed_password: str, salt: str, password: str) -> bool:
#         salted_pass = password.encode('utf-8') + salt.encode('utf-8')
#         return hashed_password == hashlib.sha512(salted_pass).hexdigest()

#     @staticmethod
#     def encode_password(password: str, salt=uuid.uuid4().hex) -> tuple[str, str]:
#         salted_pass = password.encode('utf-8') + salt.encode('utf-8')
#         hashed_password = hashlib.sha512(salted_pass).hexdigest()
#         return hashed_password, salt
