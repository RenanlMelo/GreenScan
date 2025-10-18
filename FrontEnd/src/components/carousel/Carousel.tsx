import React, { useState } from "react";
import { Dimensions, Image, Text, View } from "react-native";
import Carousel from "react-native-reanimated-carousel";
import { styles } from "./Styles";

const { width } = Dimensions.get("window");

export default function HomeCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0);

  const images = [
    {
      id: "1",
      image: require("../../../assets/images/img1.jpg"),
      title: "Learn how Plantia helps 10,000+ farmers",
      body: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
    },
    {
      id: "2",
      image: require("../../../assets/images/img2.jpg"),
      title: "Save your plants easily",
      body: "Keep your garden healthy with AI powered monitoring.",
    },
    {
      id: "3",
      image: require("../../../assets/images/img1.jpg"),
      title: "Get tips from experts",
      body: "Thousands of farmers share their best practices here.",
    },
    {
      id: "4",
      image: require("../../../assets/images/img2.jpg"),
      title: "Track your progress",
      body: "Monitor growth with Plantia’s smart dashboard.",
    },
  ];

  return (
    <View
      style={{
        height: "auto",
      }}
    >
      <Carousel
        loop
        autoPlay
        autoPlayInterval={4000}
        width={width}
        height={250}
        data={images}
        scrollAnimationDuration={800}
        onSnapToItem={(index) => setCurrentIndex(index)}
        renderItem={({ item }) => (
          <View style={styles.card}>
            <Image
              source={item.image}
              style={styles.image}
              resizeMode="cover"
            />

            <View style={styles.overlayContainer}>
              <Text style={styles.imageTitle}>{item.title}</Text>
              <Text style={styles.imageBody}>{item.body}</Text>
            </View>
          </View>
        )}
      />

      <View style={styles.dotsContainer}>
        {images.map((_, index) => (
          <View
            key={index}
            style={[styles.dot, currentIndex === index && styles.activeDot]}
          />
        ))}
      </View>
    </View>
  );
}
