"""Authentication middleware for API."""

import secrets
from typing import Optional
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from loguru import logger


security = HTTPBasic()


class AuthManager:
    """Manages API authentication."""

    def __init__(self, username: str, password: str, auth_required: bool = True):
        """
        Initialize auth manager.

        Args:
            username: Username for authentication
            password: Password for authentication
            auth_required: Whether authentication is required
        """
        self.username = username
        self.password = password
        self.auth_required = auth_required

        logger.info(f"Auth manager initialized (auth_required={auth_required})")

    def verify_credentials(
        self,
        credentials: HTTPBasicCredentials = Depends(security)
    ) -> str:
        """
        Verify HTTP Basic Auth credentials.

        Args:
            credentials: HTTP Basic credentials

        Returns:
            Username if valid

        Raises:
            HTTPException: If credentials are invalid
        """
        if not self.auth_required:
            return "anonymous"

        # Compare using secrets.compare_digest to prevent timing attacks
        correct_username = secrets.compare_digest(
            credentials.username.encode("utf8"),
            self.username.encode("utf8")
        )
        correct_password = secrets.compare_digest(
            credentials.password.encode("utf8"),
            self.password.encode("utf8")
        )

        if not (correct_username and correct_password):
            logger.warning(f"Failed authentication attempt for user: {credentials.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Basic"},
            )

        return credentials.username


# Global auth manager instance (will be initialized by app)
auth_manager: Optional[AuthManager] = None


def get_current_user(credentials: HTTPBasicCredentials = Security(security)) -> str:
    """
    Dependency to get current authenticated user.

    Args:
        credentials: HTTP Basic credentials

    Returns:
        Username
    """
    if auth_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth manager not initialized"
        )

    return auth_manager.verify_credentials(credentials)
