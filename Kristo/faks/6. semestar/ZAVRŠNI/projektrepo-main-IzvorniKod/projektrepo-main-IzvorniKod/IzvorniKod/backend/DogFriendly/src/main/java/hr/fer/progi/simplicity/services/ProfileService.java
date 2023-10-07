package hr.fer.progi.simplicity.services;

import hr.fer.progi.simplicity.entities.User;

public interface ProfileService {
    User updateAccountActivatedByEmail(long userId, boolean accountActivatedByEmail);
    User updateUsername(long userId, String newUsername);
    User updatePassword(long userId, String newPassword);
    void deleteUserById(long userId);
    void deleteUserByUsername(String username);
}
