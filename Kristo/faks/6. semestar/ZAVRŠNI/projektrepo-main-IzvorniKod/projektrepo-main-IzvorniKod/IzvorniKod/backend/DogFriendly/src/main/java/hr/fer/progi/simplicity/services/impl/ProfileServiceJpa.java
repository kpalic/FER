package hr.fer.progi.simplicity.services.impl;

import hr.fer.progi.simplicity.entities.*;
import hr.fer.progi.simplicity.repositories.UserRepository;
import hr.fer.progi.simplicity.security.CustomUserDetailsService;
import hr.fer.progi.simplicity.security.exceptions.EntityMissingException;
import hr.fer.progi.simplicity.services.ProfileService;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class ProfileServiceJpa implements ProfileService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private UserRatingServiceJpa userRatingServiceJpa;

    @Autowired
    private CustomUserDetailsService customUserDetailsService;

    @Override
    public User updateAccountActivatedByEmail(long userId, boolean accountActivatedByEmail) {
        User user = customUserDetailsService.getUserById(userId).orElseThrow(
                () -> new EntityMissingException(User.class, userId));

        user.setAccountActivatedByEmail(accountActivatedByEmail);
        return userRepository.save(user);
    }

    @Override
    public User updateUsername(long userId, String newUsername) {
        User user = customUserDetailsService.getUserById(userId).orElseThrow(
                () -> new EntityMissingException(User.class, userId));

        if (customUserDetailsService.userExistsByUserName(newUsername)) {
            throw new RequestDeniedException("Korisničko ime je zauzeto.");
        }

        user.setUsername(newUsername);
        userRepository.save(user);
        return user;
    }

    @Override
    public User updatePassword(long userId, String newPassword) {
        User user = customUserDetailsService.getUserById(userId).orElseThrow(
                () -> new EntityMissingException(User.class, userId));

        BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();
        user.setPassword(passwordEncoder.encode(newPassword));
        userRepository.save(user);
        return user;
    }

    @Override
    public void deleteUserById(long userId) {
        User user = customUserDetailsService.getUserById(userId).orElseThrow(
                () -> new EntityMissingException(User.class, userId));

        userRatingServiceJpa.deleteAllByUser(user);
        userRepository.deleteById(userId);
    }

    @Override
    public void deleteUserByUsername(String username) {
        User user = customUserDetailsService.getUserByUsername(username);

        userRatingServiceJpa.deleteAllByUser(user);
        userRepository.deleteByUsername(username);
    }
}
