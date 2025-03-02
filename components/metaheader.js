import { siteURL } from "@/constant/site";

export default function metaheader(
  titleInit = "",
  descriptionInit = "",
  keywordsInit = "",
  path = "/"
) {
  const title = titleInit
    ? titleInit
    : "Goh S.W. | A Visionary and a Passionate Entrepreneur";

  const description = descriptionInit
    ? descriptionInit
    : "Dr. Goh's investment portfolio has spread across real estate developments, technology companies and civil engineering projects globally";

  const urlPath = siteURL + path;

  return {
    metadataBase: new URL(siteURL),
    title,
    description,
    icons: {
      icon: [
        {
          rel: "icon",
          url: "/favicon.ico",
          type: "image/x-icon",
          sizes: "any",
        },
      ],
    },
    alternates: { canonical: urlPath },
  };
}
