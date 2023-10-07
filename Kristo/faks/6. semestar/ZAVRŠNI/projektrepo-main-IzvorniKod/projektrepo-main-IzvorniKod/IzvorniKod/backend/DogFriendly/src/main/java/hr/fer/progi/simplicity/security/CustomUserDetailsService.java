package hr.fer.progi.simplicity.security;

import hr.fer.progi.simplicity.entities.*;
import hr.fer.progi.simplicity.repositories.UserRepository;
import hr.fer.progi.simplicity.services.BusinessService;
import hr.fer.progi.simplicity.services.CardService;
import hr.fer.progi.simplicity.services.EmailSenderService;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.Assert;
import java.util.Date;
import java.util.Optional;

@Service
public class CustomUserDetailsService implements UserDetailsService{

    @Autowired
    private UserRepository userRepository;

    @Autowired
    BusinessService businessService;

    @Autowired
    CardService cardService;

    @Autowired
    private EmailSenderService emailSenderService;

    private static final String USERNAME_FORMAT = "^[a-zA-Z0-9]+([._]?[a-zA-Z0-9]+)*$";
    private static final String EMAIL_FORMAT = "^[a-zA-Z0-9_!#$%&'*+/=?`{|}~^.-]+@[a-zA-Z0-9.-]+$";


    @Override
    @Transactional
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = null;

        try {
            user = userRepository.findByUsername(username);
        }catch(EmptyResultDataAccessException ex) {
            throw new UsernameNotFoundException("User not found with username : " + username);
        }

        return UserPrincipal.create(user);
    }

    // This method is used by JWTAuthenticationFilter
    @Transactional
    public UserDetails loadUserById(Long id) {
        User user;

        try {
            user = userRepository.getUserById(id);
        }catch(EmptyResultDataAccessException ex) {
            throw new UsernameNotFoundException("User not found with id : " + id);
        }

        return UserPrincipal.create(user);
    }

    public User getUserByUsername(String username) {
        User user;

        try {
            user = userRepository.findByUsername(username);
        }catch(EmptyResultDataAccessException ex) {
            throw new RequestDeniedException("User not found with username : " + username);
        }

        return user;
    }

    public Optional<User> getUserById(long userId) {
        return userRepository.findById(userId);
    }

    public boolean userExistsByUserName(String username) {
        User user;

        try {
            user = userRepository.findByUsername(username);
            if(user!=null)
                return true;
        }catch(EmptyResultDataAccessException ex) {
            return false;
        }

        return false;
    }

    public boolean userExistsByEmail(String email) {
        User user;

        try {
            user = userRepository.findByEmail(email);
            if(user!=null)
                return true;
        }catch(EmptyResultDataAccessException ex) {
            return false;
        }

        return false;
    }

    public User createUser(String username, String email, String password, RoleType role) {
        BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

        //CHECK DATA
        checkUserData(username, email, password);

        //CREATE NEW USER
        User user = userRepository.save(new User(username, email, passwordEncoder.encode(password), role));

        //SEND E-MAIL - OBAVEZNO MAKNUTI TRUE IZ USER !!!!!
        emailSenderService.sendConfirmationEmail(username, email);

        return user;
    }


    public Owner createOwner(String username, String email, String password,
                            String businessName, String businessType, String businessAddress, String businessCity, String businessOIB, String businessMobileNumber, String businessDescription,
                            String cardNumber, String expiryDateMonth, String expiryDateYear, String cvv) {
        BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();
        //CHECK DATA
        checkUserData(username, email, password);
        businessService.checkBusinessData(businessName, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription);
        Date pomDate = cardService.checkCardData(cardNumber, expiryDateMonth, expiryDateYear, cvv);

        //CREATE OBJECTS
        Business business = businessService.createBusiness(businessName, businessType, businessAddress, businessCity, businessOIB, businessMobileNumber, businessDescription);
        Card card = cardService.createNewCard(cardNumber, pomDate, cvv);
        Owner owner = userRepository.save(new Owner(username, email, passwordEncoder.encode(password), RoleType.OWNER, card, business));

        //SEND E-MAIL
        emailSenderService.sendConfirmationEmail(username, email);
        emailSenderService.sendSubscriptionEmail(email);

        return owner;
    }


    public void checkUserData (String username, String email, String password){
            //USERNAME
            Assert.notNull(username, "Korisničko ime mora biti predano.");
            Assert.hasText(username, "Korisničko ime mora biti postavljeno.");
            Assert.isTrue(username.matches(USERNAME_FORMAT), "Neispravno korisničko ime.");
            if (userExistsByUserName(username)) {
                throw new RequestDeniedException("Korisničko ime '" + username + "' je zauzeto.");
            }

            //EMAIL
            Assert.notNull(email, "E-mail mora biti predan.");
            Assert.hasText(email, "E-mail mora biti postavljen.");
            Assert.isTrue(email.matches(EMAIL_FORMAT), "Neispravan e-mail.");
            if (userExistsByEmail(email)) {
                throw new RequestDeniedException("E-mail '" + email + "' je zauzet.");
            }

            //PASSWORD
            Assert.notNull(password, "Lozinka mora biti predana.");
            Assert.hasText(password, "Lozinka mora biti postavljena.");
            Assert.isTrue(password.length() >=8, "Lozinka mora sadržavati minimalno 8 znakova.");

    }
}

