package hr.fer.progi.simplicity.services;

import hr.fer.progi.simplicity.entities.Business;
import hr.fer.progi.simplicity.entities.Card;

import javax.crypto.BadPaddingException;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.Date;
import java.util.Optional;

public interface CardService {
    Date checkCardData(String cardNumber, String expiryDateMonth, String expiryDateYear, String cvv);

    Card createNewCard(String cardNumber, Date endDate, String cvv);

    String encrypt(String input, String keyStr, String ivVectorStr)
            throws NoSuchPaddingException, NoSuchAlgorithmException, InvalidAlgorithmParameterException, InvalidKeyException, IllegalBlockSizeException, BadPaddingException;

    String decrypt(String input, String keyStr, String ivVectorStr)
            throws NoSuchPaddingException, NoSuchAlgorithmException, InvalidAlgorithmParameterException, InvalidKeyException, IllegalBlockSizeException, BadPaddingException;

    Card deleteCard(String cardNumber);

    Optional<Card> getCardById (long id);
}
